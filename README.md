# WhosThatPokemon
Here we write a machine learning program to classify pictures of Pokemon.
We use machine learning Python libraries including TensorFlow and Keras.
Written by Shaun Miller using the Kaggle data set https://www.kaggle.com/thedagger/pokemon-generation-one.
This program follows some of the tutorial from the TensorFlow website: https://www.tensorflow.org/tutorials/images/classification.

The program first creates a machine learning model and trains the model with the specified data set containing hundreds of generation 1 Pokemon images.
After learning from the data set, we test the model with a realistic Pokemon image designed by Joshua Dunlop.
A full gallery of realistic images can be found on https://www.artstation.com/joshuadunlop/albums/1256278.

To run the program, first download the dependencies.
```
pip install -r requirements.txt

```

Some testing images contained in the repository include joshua-dunlop-charmander.jpg, joshua-dunlop-bulbasaur.jpg, and joshua-dunlop-aerodactyl.jpg.
To test different images, simply replace joshua-dunlop-alakazam.jpg with different file name in the following command.
Run and test the model:

```
python main.py joshua-dunlop-alakazam.jpg

```

The following PNG image is the result of a test with the image joshua-dunlop-alakazam.jpg.

![alt text](https://github.com/shaunmillerc1010/WhosThatPokemon/blob/main/preview.png)
