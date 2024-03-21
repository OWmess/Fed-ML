#!/usr/bin/env python
# coding: utf-8

# In[26]:


# third party
import matplotlib.pyplot as plt

# relative import
from mnist_dataset import mnist
from mnist_dataset import mnist_raw
import numpy as np

# syft absolute
import syft as sy

print(f"{sy.__version__ = }")


# ## 1. Launch the domain, upload the data

# In[27]:


root_client = sy.login(email='info@openmined.org', password='changethis', port=8081)


# ### Load the MNIST dataset

# Let's load the raw MNIST images and show with the `mnist_raw` function from [`mnist_datasets.py`](./datasets.py)

# In[28]:


train_images, train_labels, _, _ = mnist_raw()


# In[29]:


print("train image num: "+str(len(train_images)))
plt.imshow(train_images[0])


# In[30]:


train_labels[0]


# In[31]:


print(f"{train_images.shape = }")
print(f"{train_labels.shape = }")


# ### Processing: Flattening the MNIST images and apply one-hot encoding on the labels

# In[32]:


train_images, train_labels, _, _ = mnist()


# ### Get a subset of MNIST

# In[33]:


num_samples = 1000


# In[34]:


train_images = train_images[:num_samples, :]
train_labels = train_labels[:num_samples, :]


# In[35]:


print(f"{train_images.shape = }")
print(f"{train_labels.shape = }")


# The `train_images` and `train_labels` are the private data. Let's create similar mock data with the same shape

# In[36]:


mock_images = np.random.rand(num_samples, 784)
mock_images.shape


# In[37]:


mock_labels = np.eye(10)[np.random.choice(10, num_samples)]
mock_labels.shape


# In[38]:


assert mock_labels.shape == train_labels.shape
assert mock_images.shape == train_images.shape


# ### The DO uploads the data

# In[39]:


dataset = sy.Dataset(
    name="MNIST data",
    description="""Contains the flattened training images and one-hot encoded training labels.""",
    url="https://storage.googleapis.com/cvdf-datasets/mnist/",
)

dataset.add_contributor(
    role=sy.roles.UPLOADER,
    name="Alice",
    email="alice@openmined.com",
    note="Alice is the data engineer at the OpenMined",
)

dataset.contributors


# In[40]:


asset_mnist_train_input = sy.Asset(
    name="MNIST training images",
    description="""The training images of the MNIST dataset""",
    data=train_images,
    mock=mock_images,
)

asset_mnist_train_labels = sy.Asset(
    name="MNIST training labels",
    description="""The training labels of MNIST dataset""",
    data=train_labels,
    mock=mock_labels,
)

dataset.add_asset(asset_mnist_train_input)
dataset.add_asset(asset_mnist_train_labels)


# In[41]:


root_client.upload_dataset(dataset)


# ### The DO inspects the uploaded data

# In[42]:


datasets = root_client.api.services.dataset.get_all()
assert len(datasets) == 1
datasets


# #### The first asset of the dataset contains the training and mock images

# In[43]:


datasets[0].assets[0]


# #### The second asset contains the training and mock labels

# In[44]:


datasets[0].assets[1]


# ### The DO creates an account for the Data Scientist (DS)

# In[45]:


register_result = root_client.register(
    name="Sheldon Cooper",
    email="sheldon@caltech.edu",
    password="changethis",
    password_verify="changethis",
    institution="Caltech",
    website="https://www.caltech.edu/",
)


# In[46]:


assert isinstance(register_result, sy.SyftSuccess)


# ### 📓 Now switch to the [first DS's notebook](./01-data-scientist-submit-code.ipynb)

# In[46]:




