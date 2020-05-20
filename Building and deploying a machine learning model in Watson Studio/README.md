## Building and Deploying a machine learning model in Watson Studio

In this tutorial a machine learning model is developed Watson Studio and deployed using the Watson Machine Learning client library(And a little rant I had to do after getting angry with both tools).

### Summary

For this scenario an API is needed that is capable for predicting if a customer is going to churn to another telco.

1. To start we have to load the data, one the rigth panel you can load dataset you have as a pandas dataframe

![](/Building%20and%20deploying%20a%20machine%20learning%20model%20in%20Watson%20Studio/Images/IMG_BD_Load_Data.png)
2. Afterward it is alwasy important to do some analysis to ensure data validity, in this case we found a very skew dataset towards Churn = No therefore we will use SMOTE to oversample the data.

![](/Building%20and%20deploying%20a%20machine%20learning%20model%20in%20Watson%20Studio/Images/IMG_BD_Data_Analysis.png)

3. The heatmap of correlation is important to determine if there is any variable that is redundant

![](/Building%20and%20deploying%20a%20machine%20learning%20model%20in%20Watson%20Studio/Images/IMG_BD_Heat_Map.png)

4. The model creation is really simple thanks to scikit-learn(For now just Logistic Regression is applied)

![](/Building%20and%20deploying%20a%20machine%20learning%20model%20in%20Watson%20Studio/Images/IMG_BD_LR.png)

5. We have to connecto to Watson ML and store the mode

![](/Building%20and%20deploying%20a%20machine%20learning%20model%20in%20Watson%20Studio/Images/IMG_BD_Save_Model.png)

6. The model is deployed and ready to be used as an API

![](/Building%20and%20deploying%20a%20machine%20learning%20model%20in%20Watson%20Studio/Images/IMG_BD_Deploy_Model.png)