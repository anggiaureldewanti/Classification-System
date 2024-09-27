import pandas as pd
import numpy as np

# Fungsi untuk menghitung entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy  

# Fungsi untuk menghitung gain
def InfoGain(data, split_attribute_name, target_name="Kelas"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - weighted_entropy
    return Information_Gain

# Fungsi untuk membangun tree
def ID3(data, originaldata, features, target_attribute_name="Kelas", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        
        return tree

# Fungsi untuk memprediksi kelas dari instance baru
def predict(query, tree, default=None):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result

# Membaca data dari file CSV
data = pd.read_csv(r'D:\COOLYEAH\SEMESTER 4\kecerdasan buatan\UAS\Keadaan_Rumah.csv')

# Menentukan atribut dan target
features = ["Keadaan_Dinding_Rumah", "Keadaan_Lantai_Rumah", "Pekerjaan", "Kepemilikan_Anak_Balita_Ibu_Hamil", "Kepemilikan_Anak_Sekolah", "Kepemilikan_Lansia_Disabilitas"]
target = "Kelas"

# Periksa apakah semua kolom yang diharapkan ada dalam data
expected_columns = features + [target]
missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    print(f"Kolom yang hilang dalam data: {missing_columns}")
else:
    # Tampilkan nilai Information Gain untuk setiap atribut
    for feature in features:
        gain = InfoGain(data, feature, target)
        print(f"Information Gain for {feature}: {gain}")
    
    # Tampilkan distribusi data
    print(data.describe())
    print(data[target].value_counts())
    for feature in features:
        print(f"\nDistribusi untuk {feature}:")
        print(data[feature].value_counts())
    
    # Membangun pohon keputusan
    tree = ID3(data, data, features, target)

    # Output tree
    import pprint
    pprint.pprint(tree)

    # Input dari terminal
    print("\nMasukkan nilai atribut untuk prediksi:")
    keadaan_dinding_rumah = input("Keadaan Dinding Rumah (Bambu/Tembok): ")
    keadaan_lantai_rumah = input("Keadaan Lantai Rumah (Tanah/Keramik/Teraso): ")
    pekerjaan = input("Pekerjaan (Buruh Tani/PNS/TNI/POLRI/Petani/Pedagang): ")
    kepemilikan_anak_balita_ibu_hamil = input("Kepemilikan Anak Balita/Ibu Hamil (Ya/Tidak): ")
    kepemilikan_anak_sekolah = input("Kepemilikan Anak Sekolah (Ya/Tidak): ")
    kepemilikan_lansia_disabilitas = input("Kepemilikan Lansia/Disabilitas (Ya/Tidak): ")

    # Membuat query dari input
    query = {
        "Keadaan_Dinding_Rumah": keadaan_dinding_rumah,
        "Keadaan_Lantai_Rumah": keadaan_lantai_rumah,
        "Pekerjaan": pekerjaan,
        "Kepemilikan_Anak_Balita_Ibu_Hamil": kepemilikan_anak_balita_ibu_hamil,
        "Kepemilikan_Anak_Sekolah": kepemilikan_anak_sekolah,
        "Kepemilikan_Lansia_Disabilitas": kepemilikan_lansia_disabilitas
    }

    # Prediksi kelas untuk input
    prediction = predict(query, tree, default="Unknown")
    print(f"Prediksi Kelas: {prediction}")
