#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# third party
import jax
import matplotlib.pyplot as plt
import numpy as np

# syft absolute
import syft as sy


# ## 1. DS logins to the domain with the credentials created by the DO

# In[ ]:


node = sy.login(email='info@openmined.org', password='changethis', port=8081)
ds_client = node.login(email="sheldon@caltech.edu", password="changethis")


# ### Inspect the datasets on the domain

# In[ ]:


datasets = ds_client.datasets.get_all()
assert len(datasets) == 1
print(datasets)


# In[ ]:


assets = datasets[0].assets
assert len(assets) == 2


# In[ ]:


training_images = assets[0]
print(training_images)


# In[ ]:


training_labels = assets[1]
print(training_labels)


# #### The DS can not access the real data

# In[ ]:


assert training_images.data is None


# #### The DS can only access the mock data, which is some random noise

# In[ ]:


mock_images = training_images.mock
plt.imshow(np.reshape(mock_images[0], (28, 28)))


# #### We need the pointers to the mock data to construct a `syft` function (later in the notebook)

# In[ ]:


mock_images_ptr = training_images.pointer
print(mock_images_ptr)


# In[ ]:


type(mock_images_ptr)


# In[ ]:


mock_labels = training_labels.mock
mock_labels_ptr = training_labels.pointer
print(mock_labels_ptr)


# ## 2. The DS prepare the training code and experiment on the mock data

# In[ ]:


def mnist_3_linear_layers(mnist_images, mnist_labels):
    # import the packages
    # stdlib
    import itertools
    import time

    # third party
    from jax import grad
    from jax import jit
    from jax import random
    from jax.example_libraries import optimizers
    from jax.example_libraries import stax
    from jax.example_libraries.stax import Dense
    from jax.example_libraries.stax import LogSoftmax
    from jax.example_libraries.stax import Relu
    import jax.numpy as jnp
    import numpy.random as npr

    # define the neural network
    init_random_params, predict = stax.serial(
        Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax
    )

    # initialize the random parameters
    rng = random.PRNGKey(0)
    _, init_params = init_random_params(rng, (-1, 784))

    # the hyper parameters
    num_epochs = 10
    batch_size = 4
    num_train = mnist_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    step_size = 0.001
    momentum_mass = 0.9

    # initialize the optimizer
    opt_init, opt_update, get_params = optimizers.momentum(
        step_size, mass=momentum_mass
    )
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    def data_stream():
        """
        Create a batch of data picked randomly
        """
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield mnist_images[batch_idx], mnist_labels[batch_idx]

    def loss(params, batch):
        inputs, targets = batch
        preds = predict(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    def accuracy(params, batch):
        inputs, targets = batch
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(predict(params, inputs), axis=1)
        return jnp.mean(predicted_class == target_class)

    batches = data_stream()
    train_accs = []
    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time
        params = get_params(opt_state)
        train_acc = accuracy(params, (mnist_images, mnist_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        train_accs.append(train_acc)

    return train_accs, params


# In[ ]:


train_accs, params = mnist_3_linear_layers(
    mnist_images=mock_images, mnist_labels=mock_labels
)


# #### Inspect the training accuracies and the shape of the model's parameters

# In[ ]:


train_accs


# In[ ]:


jax.tree_map(lambda x: x.shape, params)


# ## 3. Now that the code works on mock data, the DS submits the code request for execution to the DO

# #### First the DS wraps the training function with the `@sy.syft_function` decorator

# In[ ]:


@sy.syft_function(
    input_policy=sy.ExactMatch(
        mnist_images=mock_images_ptr, mnist_labels=mock_labels_ptr
    ),
    output_policy=sy.SingleExecutionExactOutput(),
)
#三层线性神经网络
#第一层：全连接层，有1024个神经元，激活函数是ReLU。
#第二层：全连接层，有1024个神经元，激活函数是ReLU。
#第三层：全连接层，有10个神经元，用于输出每个数字的预测概率，使用LogSoftmax作为激活函数。
def mnist_3_linear_layers(mnist_images, mnist_labels):
    # import the packages
    # stdlib
    import itertools
    import time

    # third party
    from jax import grad
    from jax import jit
    from jax import random
    from jax.example_libraries import optimizers
    from jax.example_libraries import stax
    from jax.example_libraries.stax import Dense
    from jax.example_libraries.stax import LogSoftmax
    from jax.example_libraries.stax import Relu
    import jax.numpy as jnp
    import numpy.random as npr

    # define the neural network
    init_random_params, predict = stax.serial(
        Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax
    )

    # initialize the random parameters
    rng = random.PRNGKey(0)
    _, init_params = init_random_params(rng, (-1, 784))

    # the hyper parameters
    num_epochs = 10
    batch_size = 4
    num_train = mnist_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    step_size = 0.001
    momentum_mass = 0.9

    # initialize the optimizer
    opt_init, opt_update, get_params = optimizers.momentum(
        step_size, mass=momentum_mass
    )
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    def data_stream():
        """
        Create a batch of data picked randomly
        """
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield mnist_images[batch_idx], mnist_labels[batch_idx]

    def loss(params, batch):
        inputs, targets = batch
        preds = predict(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    def accuracy(params, batch):
        inputs, targets = batch
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(predict(params, inputs), axis=1)
        return jnp.mean(predicted_class == target_class)

    batches = data_stream()
    train_accs = []
    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time
        params = get_params(opt_state)
        train_acc = accuracy(params, (mnist_images, mnist_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        train_accs.append(train_acc)

    return train_accs, params


# #### Then the DS creates a new project with relevant name and description, as well as specify itself as a member of the project

# In[ ]:


new_project = sy.Project(
    name="Training a 3-layer jax neural network on MNIST data",
    description="""Hi, I would like to train my neural network on your MNIST data 
                (I can download it online too but I just want to use Syft coz it's cool)""",
    members=[ds_client],
)
new_project


# #### Add a code request to the project

# In[ ]:


new_project.create_code_request(obj=mnist_3_linear_layers, client=ds_client)


# In[ ]:


ds_client.code


# #### Start the project which will notifies the DO

# In[ ]:


project = new_project.start()


# In[ ]:


project.events


# In[ ]:


project.requests


# In[ ]:


project.requests[0]


# ### 📓 Now switch to the [second DO's notebook](./02-data-owner-review-approve-code.ipynb)

# In[ ]:




