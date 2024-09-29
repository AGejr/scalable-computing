# Import necessary modules
import os

# Define the file path
file_path = '/data/hello_world.txt'

# Write "Hello World!" to the file
try:
    with open(file_path, 'w') as file:
        file.write("Hello World!")
    print(f"'Hello World!' has been written to {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
