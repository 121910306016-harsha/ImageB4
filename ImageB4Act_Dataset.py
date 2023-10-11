#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
def read_tsv_file(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    return df
file_path = 'C:\AMC project'
df = read_tsv_file('C:/AMC project/MEDIC_train.tsv')
df.dropna(inplace=True)
print(df)


# In[2]:



filtered_df = df[df['humanitarian'] == 'affected_injured_or_dead_people']
filtered_df


# In[3]:


from collections import Counter
Counter(df['humanitarian'])


# In[4]:


Counter(filtered_df['humanitarian'])


# In[12]:


import pandas as pd
import shutil
import os
filtered_data = df[df['humanitarian'] == 'affected_injured_or_dead_people']
# Create a new directory for the ImageB4 dataset
os.makedirs('C:/Users/chint/OneDrive/Desktop/ImageB4', exist_ok=True)
p='C:/AMC project/'
# Copy the image files to the new directory
i=0
for index, row in filtered_data.iterrows():
    i=i+1
    image_path = p+row['image_path']
    image_id = row['image_id']
    new_image_path = os.path.join('C:/Users/chint/OneDrive/Desktop/ImageB4', f'{i}.jpg')
    shutil.copy(image_path, new_image_path)

# Save the filtered dataset to a new CSV file
filtered_data.to_csv('C:/Users/chint/OneDrive/Desktop/ImageB4/ImageB4_dataset.csv', index=False)  # This will save the metadata of the filtered images

print("ImageB4 dataset created.")


# In[16]:


# Clustering the affected_injured_or_dead_people category 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
data = df
categories = ['affected_injured_or_dead_people']
filtered_data = data[data['humanitarian'].isin(categories)]
categories = ['affected_injured_or_dead_people']
category_mapping = {category: index for index, category in enumerate(categories)}
filtered_data['humanitarian'] = data['humanitarian'].map(category_mapping)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_data['image_path'])
X = pd.concat([pd.DataFrame(tfidf_matrix, columns=tfidf_vectorizer.get_feature_names()), filtered_data[['humanitarian']]], axis=1)
X.dropna(inplace=True)
# Perform K-Means clustering
num_clusters = 1  # You can adjust this as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
filtered_data['cluster'] = kmeans.fit_predict(X)

# Print the resulting clusters
for cluster_id in range(num_clusters):
    cluster_data = filtered_data[filtered_data['cluster'] == cluster_id]
    print(f"Cluster {cluster_id}:\n")
    print(cluster_data[['image_path']])
    print("\n")

# Optional: You can save the cluster assignments back to your CSV file if needed
filtered_data.to_csv('clustered_data.csv', index=False)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
disaster_counts = filtered_df['disaster_types'].value_counts()
plt.figure(figsize=(10, 6))
disaster_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Disaster Types')
plt.ylabel('Number of Images')
plt.title('Number of Images in Each Disaster Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[10]:



disaster_counts = filtered_df['disaster_types'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(disaster_counts, labels=disaster_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Disaster Types')
plt.axis('equal')
plt.show()


# In[11]:


plt.hist(filtered_df['damage_severity'], bins=3, color='skyblue', alpha=0.7)
plt.xlabel('Damage Severity')
plt.ylabel('Frequency')
plt.title('Distribution of Damage Severity')
plt.xticks(rotation=45)
plt.show()


# In[13]:


import seaborn as sns

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(filtered_df['informative'], palette='Set2')
plt.xlabel('Informative')
plt.ylabel('Count')
plt.title('Distribution of Informative Label')

plt.subplot(1, 2, 2)
sns.countplot(filtered_df['humanitarian'], palette='Set2')
plt.xlabel('Humanitarian')
plt.ylabel('Count')
plt.title('Distribution of Humanitarian Label')

plt.tight_layout()
plt.show()


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
damage_counts = filtered_df['damage_severity'].value_counts()
damage_order = filtered_df['damage_severity'].unique()
plt.figure(figsize=(10, 6))
sns.barplot(x=damage_counts.index, y=damage_counts.values, order=damage_order, palette='Set2')
plt.xlabel('Damage Severity')
plt.ylabel('Number of Images')
plt.title('Number of Images for Each Damage Severity Level')
plt.xticks(rotation=45)
plt.show()



# In[22]:


plt.figure(figsize=(12, 6))
sns.countplot(x='disaster_types', hue='damage_severity', data=filtered_df, palette='viridis')
plt.xlabel('Disaster Types')
plt.ylabel('Count')
plt.title('Damage Severity by Disaster Type')
plt.xticks(rotation=45)
plt.legend(title='Damage Severity')
plt.show()


# In[25]:


pip install labelImg


# In[30]:


get_ipython().system('git clone https://github.com/tzutalin/labelImg.git')


# In[31]:


get_ipython().run_line_magic('cd', 'labelImg')


# In[32]:


get_ipython().system('pip install pyqt5 lxml')


# In[34]:


get_ipython().system('pyrcc5 -o libs/resources.py resources.qrc')


# In[37]:


get_ipython().system('python labelImg.py')


# In[38]:


get_ipython().system('pip install opencv-python-headless')
get_ipython().system('pip install matplotlib')


# In[42]:


import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
image_path = 'C:/AMC project/data/aidr_disaster_types/chennai_flood/4_12_2015/672669572642635776_0.jpg'
annotation_path = 'C:/AMC project/data/aidr_disaster_types/672669572642635776_0.xml'
image = cv2.imread(image_path)
tree = ET.parse(annotation_path)
root = tree.getroot()
for obj in root.findall('object'):
    label = obj.find('name').text
    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

