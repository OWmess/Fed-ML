#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# syft absolute
import syft as sy
from syft.service.request.request import RequestStatus

print(f"{sy.__version__ = }")


# In[ ]:


node = sy.orchestra.launch(name="mnist-domain", dev_mode=True)
root_client = node.login(email="info@openmined.org", password="changethis")


# ## 1. DO reviews the submitted project and code

# In[ ]:


root_client.projects


# In[ ]:


requests = root_client.projects[0].requests
requests


# In[ ]:


assert len(requests) == 1


# In[ ]:


request = requests[0]
assert request.status == RequestStatus.PENDING
request


# In[ ]:


change = request.changes[0]
change


# #### Inspecting the submitted code

# In[ ]:


# gettting a reference to the user code object
user_code = change.code

# viewing the actual code submitted for request
user_code.show_code


# #### The data assets corresponds with the submitted code

# In[ ]:


assert len(user_code.assets) == 2
user_code.assets


# In[ ]:


mock_images = user_code.assets[0].mock
print(f"{mock_images.shape = }")
mock_labels = user_code.assets[1].mock
print(f"{mock_labels.shape = }")


# #### The DO runs the code on mock data to ensure things are fine

# In[ ]:


users_function = user_code.unsafe_function
users_function


# In[ ]:


mock_train_accs, mock_params = users_function(
    mnist_images=mock_images, mnist_labels=mock_labels
)


# In[ ]:


assert isinstance(mock_train_accs, list)
mock_train_accs


# In[ ]:


assert isinstance(mock_params, list)
mock_params


# ## 2. DO runs the submitted code on private data, then deposits the results to the domain so the DS can retrieve them

# In[ ]:


# private data associated with the asset
private_images = user_code.assets[0].data
print(f"{private_images.shape = }")
private_labels = user_code.assets[1].data
print(f"{private_labels.shape = }")


# In[ ]:


train_accs, params = users_function(
    mnist_images=private_images, mnist_labels=private_labels
)


# In[ ]:


assert isinstance(train_accs, list)
train_accs


# In[ ]:


assert isinstance(params, list)
params


# In[ ]:


res = request.accept_by_depositing_result((train_accs, params))


# In[ ]:


assert isinstance(res, sy.SyftSuccess)
res


# ### 📓 Now switch to the [second DS's notebook](./03-data-scientist-download-results.ipynb)

# In[ ]:




