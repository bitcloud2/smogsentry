# SmogSentry - Predicts a car's air pollution score

Jeffrey Roach, December 2015

Email: bitcloud22@gmail.com

LinkedIn: https://www.linkedin.com/in/jeffreyroach22

### Description
It was recently discovered that Volkswagen installed software in their diesel cars that allow them to cheat on the air emissions test. The EPA gives scores between 1 and 10 based on these tests, and low emissions mean high scores. In light of the Volkswagen scandal, which cars have accurate test results and which ones don’t? That’s where SmogSentry comes in.

SmogSentry uses a car’s specifications such as weight, horsepower, and miles per gallon to predict the car’s air pollution score. 
Most of the data came from the EPA’s green guide for 2015, while the rest of the car specifications came from a web scrape of the popular car website, Motortrend. 

### Modeling
Using nearly 1000 cars, I trained a Gradient Boosted Random Forest Classifier. This multi-class classifier needs to predict the exact score between the range of 1 and 10 in order to be considered correct. Using K-fold validation, it was determined the model was correct 90% of the time in terms of accuracy, recall, and precision. These high predictive metrics make the model well suited for a screening test.

### Webapp
SmogSentry has been integrated into a web app allowing users to select the specifications for a car they are building, or for a car the EPA might think is too good to be true.
The user can then predict what their score would be under both the Federal and California agencies.

### Results
Now, the real question is, could SmogSentry have identified past problem vehicles like the ones in the Volkswagen scandal?
Well, the Volkswagen scandal involves diesel cars, and since diesel cars are rare it’s difficult to build a robust model, so I decided to look at the gasoline versions of these cars. To make sure that I was not picking on just Volkswagen, I choose to look at all gasoline cars that had a diesel version regardless of who made them.

Of the 65 cars tested, only four cars were predicted different from what was reported to the EPA. Cars with a predicted score worse than their reported score have a negative value in the difference column. The cars with stars next to their name their diesel counterparts were included in the Volkswagen scandal. Notice that their scores are predicted much worse than what they reported to the EPA.

 To give you a sense of why these results are important, I’ve included a plot showing the differences between the predicted and reported scores. You can see that anomalies are rare and my model is very good at identifying them. Both of the cars that were predicted four ranks worse, are actually the only cars there as both models have two different engines. Fortunately, for Volkswagen it isn’t all bad news, their Jetta Hybrid was predicted to perform better than they actually reported.
As you can see, SmogSentry can be easily used as a screening test for the EPA and might lift some of the pressure from wondering which cars have accurate scores and which don’t.

