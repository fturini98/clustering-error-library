import numpy as np
import json
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
################################################## 
class error_class:
    ''' oltre a come normalizzo gli errori  devo guardare anche come metterli insieme, prodotto?, somma?, somma in quadratura? '''
    def __init__(self,type):
        self.type=type
    
    def evlauate(self,raw_data):
        if self.type=='default':
            return self.default(raw_data)
    
    def default(self,raw_data):
        #print(raw_data[0])
        points=np.array(raw_data[0])
        #print(type(points))
        errors=np.array(raw_data[1])
        true_points=np.array(raw_data[2])
        true_error=np.array(raw_data[3])
        distance0_error=np.array([np.sqrt(np.sum(np.power(error,2))) for error in errors])
        #print('error nuovo:',distance0_error)
        distance0_error=np.nan_to_num(distance0_error, nan=0.0)
        
        masses=np.where(distance0_error==0,1,1/distance0_error)
        #print(type(masses[0]))
        #print('calcolo masse',[points,masses,errors,true_points,true_error])
        
        return points,masses,errors,true_points,true_error


    
class data_class:
    
    def __init__(self,formatting_error='default'):
        self.save_memory=True
        self.formatting_error=error_class(formatting_error)
    
    def save_dictionary_to_json(tree,file_path):
    
        # Serialize the tree to JSON
        def serialize_tree(node):
            """Serialize the tree to JSON format recursively.

            This function serializes the tree object to JSON format. It handles dictionary nodes and pandas DataFrame nodes separately.

            Args:
                node: The current node in the tree.

            Returns:
                Serialized node in JSON format.
            """
            if isinstance(node, dict):
                return {k: serialize_tree(v) for k, v in node.items()}
            elif isinstance(node, pd.DataFrame):
                return {"type": "DataFrame", "data": node.to_dict(orient="split")}
            return node

        serialized_tree = serialize_tree(tree)

        # Save the serialized tree to a file
        with open(file_path, "w") as file:
            json.dump(serialized_tree, file)
    
    def import_dictionary_from_json(self,file_path):
        
        with open(file_path, "r") as file:
            serialized_tree = json.load(file)
        
        # Deserialize the tree, including DataFrames
        def deserialize_tree(node):
            """Deserialize the tree from the serialized format.

            This function deserializes the tree object from the serialized format. It handles dictionary nodes and serialized DataFrames separately.

            Args:
                node: The current node in the serialized tree.

            Returns:
                Deserialized node.
            """
            if isinstance(node, dict):
                if "type" in node and node["type"] == "DataFrame":
                    # Convert the serialized DataFrame back to a DataFrame object
                    data = node["data"]["data"]
                    index = node["data"]["index"]
                    columns = node["data"]["columns"]
                    return pd.DataFrame(data, index=index, columns=columns)
                return {k: deserialize_tree(v) for k, v in node.items()}
            return node

        tree = deserialize_tree(serialized_tree)
        
        return tree
        
    def import_simple_df_from_json(self,file_path):
          self.df=self.import_dictionary_from_json(file_path)
    
    def import_df(self,df):
        self.df=df
      
    def select_df_key(self, key_tuples):
        '''
        This method takes as input an array of tuples, where each tuple contains the tuple (key, key error), and generates a tuple
        of two arrays where one contains a series of tuples where each one represents the position of the data point, and the second array
        a list of tuples where each one represents the error of the array.
        
        Args:
            key_tuples: array of tuples, where each tuple contains the tuple (key, key error)
            
        Returns:
            tuple: A tuple containing two arrays. The first array contains the position of the data points, and the second array
                   contains the corresponding errors.
        '''  
        #key_tuples = np.array(key_tuples)
        real_positions=[]
        real_errors=[]
        positions = []
        errors = []
        
        #Elimino i dati che hanno nan in alcuni valori
        for key, key_error in key_tuples:
            self.df = self.df.dropna(subset=[key])
            
        for key, key_error in key_tuples:
            try:
                #Save in a local variable the list of array defing each feature
                position = self.df[key]
                real_positions.append(position)
                position= (position-np.mean(position))/np.std(position)#Rinormalizazione  che mantien dispersione
                positions.append(position)
                
                #Evaluation of error over mean and std deviation
                #mean_error=
                
                if key_error in [None,0,'None','0']:
                    real_error = [0 for _ in position]
                    error = real_error
                else: 
                    real_error=self.df[key_error]
                    error = (real_error)/np.std(position)
                
                errors.append(error)
                real_errors.append(real_error)
                
            except KeyError:
                print(f"Key '{key}' not found in dataframe index.")
        
        self.min_errors=[np.min(error_array) for error_array in errors]
        #self.normailize_positions(positions)
        #Associate each set of fature togheter defing an array of vector point
        real_errors=list(zip(*real_errors))
        real_positions=list(zip(*real_positions))
        positions=list(zip(*positions))
        errors=list(zip(*errors))
            
        #Save a tuple conting the vectors of each point whit the vector errors
        self.raw_data = (np.array(positions), np.array(errors),np.array(real_positions),np.array(real_errors))  # Salva la tupla in data
    
    def format_error(self):
        #print('raw:',self.raw_data)
        self.data=self.formatting_error.evlauate(self.raw_data)
        #print('data',self.data)
        #Clear df memory
        if self.save_memory:
            self.raw_data=None
        
#################################################################################

class Clusters:
    
    def __init__(self,data):

        self.array=np.array(data.data[0])
        self.masses=np.array(data.data[1])
        
        self.center_of_gravity=np.sum([mass * array for mass, array in zip(self.masses, self.array)])/(np.sum(self.masses))
        
        self.save_distances_matrix=False
        #print('Min errors:',data.min_errors)
        self.min_errors=np.array(data.min_errors,dtype=np.float64)
        self.S=10.
        
        self.fattore_scala_stop=1.
        #print('lungezza',len(self.masses))
        self.cut_val=0.0

            
        
    def calculate_potentials(self):

        distances_matrix=cdist(self.array, self.array, metric='euclidean')
        #print('Min_errors:',self.min_errors)
        self.controll_min_distances()
        
        self.min_distance=np.linalg.norm(self.min_errors)
        
        distances_matrix_without_0=np.where(distances_matrix > self.min_distance,distances_matrix,self.min_distance)
        #print('Min_errors:',self.min_errors)
        #print('Min_dist:',self.min_distance)
        #print(distances_matrix_without_0)
        inverse_distances_matrix=1/distances_matrix_without_0

        self.potentials=-np.dot(inverse_distances_matrix- np.diag(np.diag(inverse_distances_matrix)),self.masses)
        
        #if self.save_distances_matrix==True:

        self.distances_matrix=distances_matrix
        
        return distances_matrix
    
    def controll_min_distances(self):
        if 0 in self.min_errors:
            zero_indices = np.where(self.min_errors == 0)[0]
            
            for index in zero_indices:
                feature_array=[x[index]for x in self.array]
                feature_array = np.array(feature_array).reshape(-1,1)
                feature_distances_matrix=cdist(feature_array, feature_array, metric='euclidean')
                
                masked_distances = np.ma.masked_where(feature_distances_matrix == 0, feature_distances_matrix)
                min_distances = np.nanmin(masked_distances, axis=1)
                mean_min_distances=np.mean(min_distances)

                self.min_errors[index]=np.float64(mean_min_distances/self.S)
     
    ############################### Parent#################################                   
    def find_single_parent(self,distance_matrix,index):
        if index==self.root_index:
            return self.root_index
        distace_array=distance_matrix[index]
        parent_candidates_indices = np.where(self.potentials < self.potentials[index])[0]
        parent_candidates=distace_array[parent_candidates_indices]
        parent=parent_candidates_indices[np.argmin(parent_candidates)]
        return parent
        
        
    def find_parents(self):  
        distance_matrix=self.calculate_potentials()
        #print(self.potentials)
        sorted_index=np.argsort(self.potentials)
        
        self.root_index=sorted_index[0]
        
        self.parents=np.array([self.find_single_parent(distance_matrix,index) for  index in range(len(self.potentials))])
        self.weigths=np.array([distance_matrix[index][parent] for index,parent in enumerate(self.parents)])
        self.weigths[self.root_index]=np.max(self.weigths)+1
        #print(sorted_index,self.parents)
    
        
    def build_dendrogram_parent(self):
        # Inizializzazione dell'array contenente le etichette dei cluster per ciascun punto
        iteration_points_label = np.zeros(len(self.weigths))
        
        # Lista delle etichette per ogni iterazione
        points_label_cluster = [iteration_points_label.copy()]

        # Array degli indici degli elementi ordinati in base ai pesi
        sorted_index_weights = np.argsort(self.weigths)
        #sorted_index_potential = np.argsort(self.potentials)
        mean_sorted_weights=np.mean(self.weigths)
        
        # Numero relativo del cluster
        relative_cluster_number = 0

        # Iterazione sugli elementi ordinati
        for i in range(len(sorted_index_weights)):
            point_to_connect = sorted_index_weights[i]
            
            #codizione di uscita dal clustering
            if self.weigths[point_to_connect]>(mean_sorted_weights*self.fattore_scala_stop):
                break
            
            parent_point = self.parents[point_to_connect]
            point_label = iteration_points_label[point_to_connect]
            parent_label = iteration_points_label[parent_point]

            # Gestione dei diversi casi
            if point_label == 0 and parent_label == 0:
                relative_cluster_number += 1
                iteration_points_label[point_to_connect] = relative_cluster_number
                iteration_points_label[parent_point] = relative_cluster_number
            elif point_label == 0 and parent_label != 0:
                iteration_points_label[point_to_connect] = parent_label
            elif point_label != 0 and parent_label == 0:
                iteration_points_label[parent_point] = point_label
            elif point_label != 0 and parent_label != 0:
                iteration_points_label = np.where(iteration_points_label == point_label, parent_label, iteration_points_label)

            # Aggiorna i log della clusterizzazione
            points_label_cluster.append(iteration_points_label.copy())

        #matrice dei cluster per ogni iterazione
        self.points_label_cluster = points_label_cluster
        
        self.cluster_ids=iteration_points_label
        
        unique_clusters, counts = np.unique(self.cluster_ids, return_counts=True)
        
        mean_member=np.mean(counts[1:])
        for cluster_id,member in zip(unique_clusters,counts):
            if member<mean_member*self.cut_val:
                #print('Elimino cluster')
                self.cluster_ids[self.cluster_ids == cluster_id] = 0 
        
        # Calcolo dei conteggi per ogni etichetta di cluster
        unique_labels, counts = np.unique(self.cluster_ids, return_counts=True)
        #print('Prima mappatura',unique_labels, counts)
        #self.cluster_ids_non_mappato=self.cluster_ids
        
        # Ordino i cluster in ordine decrescente di popolosità (escluso il cluster 0)
        index_most_popolous_clusters = np.argsort(counts[unique_labels != 0])[::-1]
        unique_labels_sorted = unique_labels[unique_labels != 0][index_most_popolous_clusters]

        # Creiamo una mappatura delle etichette non-zero a valori negativi (poi convertiti in positivi)
        mapping = {label: i + 1 for i, label in enumerate(unique_labels_sorted)}

        # Manteniamo il cluster 0 e applichiamo la mappatura agli altri
        self.cluster_ids = np.array([mapping.get(x, 0) if x != 0 else 0 for x in self.cluster_ids])
        
                   
        #calcolo il wcss per la configurazione
        self.calculate_wcss()
        
        ####################### calcolo la media della silouette #######################
        valid_indices = self.cluster_ids != 0
        filtered_data = self.array[valid_indices]
        filtered_ids = self.cluster_ids[valid_indices]

        # Calcola la silhouette score media per tutti i punti filtrati
        self.average_silhouette=silhouette_score(filtered_data, filtered_ids)
        
        
        return self.cluster_ids
    
################################ Sons ######################################
    def find_single_son(self,distance_matrix,index):
        if index==self.sons_root_index:
            return index
        distace_array=distance_matrix[index]
        sons_candidates_indices = np.where(self.potentials > self.potentials[index])[0]
        sons_candidates=distace_array[sons_candidates_indices]
        son=sons_candidates_indices[np.argmin(sons_candidates)]
        return son

    #La funzione sopra dovrebbe andare, questa sotto non so
    def find_sons(self):
        distance_matrix=self.calculate_potentials()
        
        sorted_index=np.argsort(self.potentials)
        
        self.sons_root_index=sorted_index[-1]
        
        self.sons=np.array([self.find_single_son(distance_matrix,index) for  index in range(len(self.potentials))])
        self.sons_weigths=np.array([distance_matrix[index][son] for index,son in enumerate(self.sons)])
        self.sons_weigths[self.sons_root_index]=np.max(self.sons_weigths)+1
        
    def build_dendrogram_son(self):
        # Inizializzazione dell'array contenente le etichette dei cluster per ciascun punto
        iteration_points_label = np.zeros(len(self.sons_weigths))
        
        # Lista delle etichette per ogni iterazione
        points_label_cluster = [iteration_points_label.copy()]

        # Array degli indici degli elementi ordinati in base ai pesi
        #sorted_index_potential = np.argsort(self.sons_weigths)
        sorted_index_potential = np.argsort(self.potentials)
        
        # Numero relativo del cluster
        relative_cluster_number = 0

        # Iterazione sugli elementi ordinati
        for i in range(len(sorted_index_potential)):
            point_to_connect = sorted_index_potential[i]
            son_point = self.sons[point_to_connect]
            point_label = iteration_points_label[point_to_connect]
            son_label = iteration_points_label[son_point]

            # Gestione dei diversi casi
            if point_label == 0 and son_label == 0:
                relative_cluster_number += 1
                iteration_points_label[point_to_connect] = relative_cluster_number
                iteration_points_label[son_point] = relative_cluster_number
            elif point_label == 0 and son_label!=0:
                iteration_points_label[point_to_connect] = son_label
            elif point_label != 0 and son_label == 0:
                iteration_points_label[son_point] = point_label
            elif point_label != 0 and son_label != 0:
                iteration_points_label = np.where(iteration_points_label == point_label, son_label , iteration_points_label)

            # Aggiorna i log della clusterizzazione
            points_label_cluster.append(iteration_points_label.copy())

        self.points_label_cluster = points_label_cluster
        
        # Calcolo dei conteggi per ogni etichetta di cluster
        unique_labels, counts = np.unique(iteration_points_label, return_counts=True)
        
        # Stampa dei conteggi
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} elementi")
    
        return iteration_points_label

    def calculate_wcss(self):
        #print('ci entro')
        # Filtrare i punti che non appartengono al cluster con id 0
        mask = self.cluster_ids != 0
        filtered_array = self.array[mask]
        filtered_masses = self.masses[mask]
        filtered_clusters_id = self.cluster_ids[mask]
        
        wcss = 0.0
        
        # Ottieni gli ID unici dei cluster (escludendo lo 0)
        unique_clusters = np.unique(filtered_clusters_id)
        
        for cluster_id in unique_clusters:
            # Filtra i punti appartenenti al cluster corrente
            cluster_mask = filtered_clusters_id == cluster_id
            cluster_points = filtered_array[cluster_mask]
            cluster_masses = filtered_masses[cluster_mask]
            
            # Calcola il centroide del cluster usando le masse come pesi
            centroid = np.average(cluster_points, axis=0, weights=cluster_masses)
            
            # Calcola la somma pesata delle distanze al quadrato dei punti rispetto al centroide
            distances_squared = np.sum((cluster_points - centroid) ** 2, axis=1)
            weighted_distances_squared = distances_squared * cluster_masses
            
            # Aggiungi al WCSS totale
            wcss += np.sum(weighted_distances_squared)
        
        self.wcss=wcss


    def calculate_silhouette(self):
        # Numero di punti
        n = len(self.cluster_ids)
        
        # Inizializza i coefficienti di silhouette
        silhouette_scores = np.zeros(n)
        
        # Crea una matrice delle distanze intra-cluster
        # Creiamo un array per memorizzare la distanza media intra-cluster per ogni punto
        #intra_cluster_distances = np.zeros(n)
        
        # Creiamo una matrice che contiene i punti per cluster
        cluster_ids,counts = np.unique(self.cluster_ids, return_counts=True)
        
        # Calcola la distanza media intra-cluster per ogni punto
        for cluster_id,count in zip(cluster_ids,counts):
            if cluster_id == 0:
               continue  # Salta il cluster con ID 0
            
            cluster_mask = self.cluster_ids == cluster_id
            cluster_points = self.distances_matrix[cluster_mask][:, cluster_mask]
            
            # La distanza intra-cluster media per ogni punto
            a_values = np.mean(cluster_points, axis=1)
            #intra_cluster_distances[cluster_mask] = intra_distances
        
            # Calcola la distanza media inter-cluster per ogni punto(b)
            '''
        for cluster_id,count in zip(cluster_ids,counts):
            if cluster_id == 0:
                continue  # Salta il cluster con ID 0
            
            cluster_mask = self.cluster_ids == cluster_id
            '''  
        
            
            b_values = np.full(count, np.inf)
            for cluster_id1,count1 in zip(cluster_ids,counts):
                if cluster_id1 !=0 and cluster_id1!=cluster_id:
                    mask_other_cluster= self.cluster_ids==cluster_id1
                    distances_inter_cluster = self.distances_matrix[cluster_mask][:, mask_other_cluster]
                    avg_distances = np.mean(distances_inter_cluster, axis=1)
            
                    b_values = np.minimum(b_values, avg_distances)
            #other_clusters_mask = ~cluster_mask
            
            # Matrice delle distanze tra punti del cluster corrente e punti di altri cluster
            #distances_inter_cluster = self.distances_matrix[cluster_mask][:, other_clusters_mask]
            
            # Calcolare la distanza media al cluster più vicino
            #min_distances_inter_cluster = np.min(np.mean(distances_inter_cluster, axis=1), axis=0)
            
            silhouette_scores[cluster_mask] = (b_values - a_values) / np.maximum(a_values, b_values)
        
        self.silhouette= silhouette_scores

    def plot_silhouette(self,ax='None'):
        # Escludi i punti non clusterizzati (cluster_id == 0)
        valid_indices = self.cluster_ids != 0
        filtered_data = self.array[valid_indices]
        filtered_ids = self.cluster_ids[valid_indices]

        # Calcola la silhouette score per ogni punto filtrato
        silhouette_vals = silhouette_samples(filtered_data, filtered_ids)

        # Calcola la silhouette score media per tutti i punti filtrati
        silhouette_avg = silhouette_score(filtered_data, filtered_ids)
        print(f"Silhouette score medio: {silhouette_avg}")

        # Visualizza il risultato
        if ax=='None':
            fig, ax = plt.subplots()
        y_lower = 10
        for i in np.unique(filtered_ids):
            # Ottieni i punti appartenenti a questo cluster
            cluster_silhouette_vals = silhouette_vals[filtered_ids == i]
            
            # Ordina i valori di silhouette per questo cluster
            cluster_silhouette_vals.sort()
            
            # Calcola la dimensione del cluster
            cluster_size = cluster_silhouette_vals.shape[0]
            
            # Disegna i valori di silhouette
            ax.fill_betweenx(np.arange(y_lower, y_lower + cluster_size),
                            0, cluster_silhouette_vals,
                            alpha=0.7)
            
            # Aggiungi etichette
            ax.text(-0.05, y_lower + 0.5 * cluster_size, f'Cluster {i}',
                    color='black', ha='center', va='center')
            
            # Incrementa la posizione di y_lower
            y_lower += cluster_size

        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_xlabel("Coefficiente di Silhouette")
        ax.set_ylabel("Cluster")
        ax.set_yticks([])
        ax.set_title("Diagramma dei coefficienti di silhouette per i cluster")

