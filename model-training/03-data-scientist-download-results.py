#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# third party
import jax
from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense
from jax.example_libraries.stax import LogSoftmax
from jax.example_libraries.stax import Relu
import jax.numpy as jnp
from mnist_dataset import mnist

# syft absolute
import syft as sy


# In[ ]:


node = sy.login(email='info@openmined.org', password='changethis', port=8081)
ds_client = node.login(email="sheldon@caltech.edu", password="changethis")


# ## After the DO has ran the code and deposited the results, the DS downloads them

# In[ ]:


datasets = ds_client.datasets.get_all()
assets = datasets[0].assets
assert len(assets) == 2


# In[ ]:


training_images = assets[0]
training_labels = assets[1]


# In[ ]:


ds_client.code


# In[ ]:


result = ds_client.code.mnist_3_linear_layers(
    mnist_images=training_images, mnist_labels=training_labels
)


# In[ ]:


train_accs, params = result.get_from(ds_client)


# In[ ]:


assert isinstance(train_accs, list)
train_accs


# In[ ]:


assert isinstance(params, list)
jax.tree_map(lambda x: x.shape, params)


# ## Having the trained weights, the DS can do inference on the its MNIST test dataset

# In[ ]:


_, _, test_images, test_labels = mnist()


# In[ ]:


assert test_images.shape == (10000, 784)
assert test_labels.shape == (10000, 10)


# #### Define the neural network and the accuracy function

# In[ ]:


init_random_params, predict = stax.serial(
    Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax
)


# In[ ]:


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


# #### Test inference using random weights

# In[ ]:


rng = random.PRNGKey(0)
_, random_params = init_random_params(rng, (-1, 28 * 28))

test_acc = accuracy(random_params, (test_images, test_labels))
print(f"Test set accuracy with random weights = {test_acc * 100 : .2f}%")


# #### Test inference using the trained weights recevied from the DO

# In[ ]:


test_acc = accuracy(params, (test_images, test_labels))
print(f"Test set accuracy with trained weights = {test_acc * 100 : .2f}%")


# In[ ]:




