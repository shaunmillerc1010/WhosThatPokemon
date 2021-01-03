# WhosThatPokemon
Here we write a machine learning program to classify pictures of Pokemon.
We use machine learning Python libraries including TensorFlow and Keras.
Written by Shaun Miller using the Kaggle data set https://www.kaggle.com/thedagger/pokemon-generation-one.
This program follows some of the tutorial from the TensorFlow website: https://www.tensorflow.org/tutorials/images/classification.

The program first creates a machine learning model and trains the model with the specified data set containing hundreds of generation 1 Pokemon images. We limit the data set to 5 popular Pokemon (Mewtwo, Pikachu, Charmander, Squirtle, and Bulbasaur) for resource-preserving and instructional purposes. Instructions for including the entire 149 pokemon in the model lies at the bottom of this README.

After learning from the data set, we test the model with a realistic Pokemon image designed by Joshua Dunlop. I figured these would be great images for testing since the data set the model uses to train contains few (if any) realistic Pokemon images.
A full gallery of realistic images can be found on https://www.artstation.com/joshuadunlop/albums/1256278.
Given a specified testing image, the machine learning model makes a prediction of the Pokemon's name while displaying the model's confidence level.

To run the program, first download the dependencies.
```
pip install -r requirements.txt
```

To test different images, simply replace joshua-dunlop-pikachu.jpg with different file name in the following command (or just add it to the end).
Run and test the model:

```
python main.py joshua-dunlop-pikachu.jpg
```

The following PNG image is the result of a test with the image joshua-dunlop-pikachu.jpg and joshua-dunlop-mewtwo.jpg.

![alt text](https://github.com/shaunmillerc1010/WhosThatPokemon/blob/main/preview_pikachu.png)
![alt text](https://github.com/shaunmillerc1010/WhosThatPokemon/blob/main/preview_mewtwo.png)


Notes: Background seems to make a HUGE difference when the machine learning model makes a prediction. For example, since the model trained on Mewtwo images that had noticeably darker backgrounds, the model seems more likely to predict Mewtwo for realistic images with dark backgrounds. If you use cartoon-like images, the model seems much more accurate than when using the realistic images.


To include the larger data set, uncomment the beginnings of lines 81,82, and 85 in main.py. Then change line 127 from
```
train_ds, val_ds = get_local_data('dataset_popular')
```
to
```
train_ds, val_ds = get_local_data('dataset')
```

Warning: This will take considerable longer to train the model.
