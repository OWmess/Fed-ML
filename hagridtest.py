import syft as sy

domain = sy.login(email='info@openmined.org', password='changethis', port=8081)

# print(domain.store)

# print(domain.requests)

print(domain.api)