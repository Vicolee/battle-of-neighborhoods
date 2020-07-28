
# Best places to setup Chinese Restaurants in New York

## 1.  Background
In 2018, according to the International Monetary Fund, China's economy produced $25.3 trillion. This figure is based on the purchasing power parity which takes into account the effect of exchange rates, making it the best method for comparing gross domestic product by country.

This makes China the world's largest economy, seconded by the EU with $22 trillion and the United States third, producing $20.5 trillion.

Moreover, in terms of demographics in the United States, the population of Chinese immigrants have grown nearly seven-fold since 1980, reaching almost  2.5 million in 2018 (5.5 percent of the overall foreign-born population) [[source]]([https://www.migrationpolicy.org/article/chinese-immigrants-united-states]). With such an astounding rate of growth of the Chinese community in the United States, it comes with no surprise that there has been an explosion in the demand of Chinese cuisines.

In this project, I will be using Machine Learning classifiers like KNN and Logistic Regression to predict, based on a venue's surrounding neighborhood, whether the location is a good place to setup a Chinese restaurant in New York.

## 2. Problem

Data that might contribute to determining the success of a Chinese Restaurant might include factors like its location and surrounding venues like cinemas. In this project, I will be using Machine Learning Classifiers, like KNN and Logistic Regression, to predict whether a venue in New York will be a good location for setting up a chinese restaurant.

## 3. Data

### 3.1 Data Sources
The data for the coordinates for each neighborhood and borough in New York can be found [here](https://ibm.box.com/shared/static/fbpwbovar7lf8p5sgddm06cgipa2rxpe.json). With these coordinates, I used the [Foursquare API](https://developer.foursquare.com/docs/places-api/endpoints/)  to retrieve all Chinese Restaurants in each neighborhood in New York. Using the same API, with the method [explore](https://developer.foursquare.com/docs/venues/explore), I retrieved the  venues within 2km radius from each Chinese Restaurant. The metric used to train the Classifiers is 'Popularity' of the restaurant. However, since this was not readily available, I had to retrieve the number of 'Likes' for each restaurant using the [venues](https://developer.foursquare.com/docs/api-reference/venues/details) method in the Foursquare API.

### 3.2 Data Cleaning
The above mentioned datasets had several problems and needed to be cleansed before fitting into the classifiers.

Firstly, the New York data did not come with data for the chinese restaurants. Thus, I needed to use the foursquare api to scrape the existing chinese restaurants in each neighborhood.

Note that the data for this project is limited in the sense that the details of each venue gathered are based on foursquare's existing database.

![ny_df](images/ny_df)

As the foursquare api could only retrieve details of restaurants based on the radius from the centre of a specified latitude and longitude, I was unable to retrieve all chinese restaurants by neighborhood. Moreover, due to limitations in the number of calls I could do as I was a non-premium member, I limited the radius from the center of the neighborhood to 2km. Thus, only chinese restaurants which are within a 2km radius from the center of the neighborhood would be retrieved from the foursquare api.

See below for the resulting dataframe, which I have named chinese_restaurant_df:

![chinese_restaurant_df_unfiltered](images/chinese_restaurant_df_unfiltered)

After exploring chinese_restaurant_df, I noticed that there was a discrepancy in the number of unique neighborhoods (300) as opposed to the number of unique chinese restaurants by coordinates (270). Moreover, while the number of unique chinese restaurants by name is 255, the unique count by restaurant ID is 270. Thus, it is likely that some chinese restaurants have the same name, yet are not referring to the same restaurants since they have different IDs and are located in different places.

![chinese_restaurant_df_exp](images/chinese_restaurant_df_exp)

Now, I will create a new unique_chinese_restaurant_df by removing the duplicated restaurants by ID. The number of duplicated restaurants removed is 34, resulting in only 270 chinese restaurants in the dataframe.

Since the remaining dataframe consists only of unique chinese restaurants, I will now continue to gather more data about each of these restaurants to prepare for training. Using the foursquare api's [venues method](https://developer.foursquare.com/docs/api-reference/venues/details) I will retrieve the 'Likes', 'Ratings', and 'Price Tier' of each chinese restaurant. Foursquare API defines 'Price Tier' as a value ranging from 1 (least pricey) - 4 (most pricey) for each venue.

The new dataframe now looks like the below:

![chinese_restaurant_details_df](images/chinese_restaurant_details_df)

Exploring the dataframe, we get:

![chinese_restaurant_df_info](images/chinese_restaurant_df_info)

Notice that the column 'Ratings' has 99 non-null Count which means more than 60% of the values for the column is missing. Thus, I will be dropping the entire column before feeding it for training as it would not provide much information.

Next, I will normalize the values of 'Likes', 'Ratings', and 'Price Tier' to plot a bar graph so as to visualize the top 10 most popular chinese restaurants by 'Likes'.

![most_popular_chinese_restaurant](images/most_popular_chinese_restaurant)

As seen earlier, there are numerous null values in the 'Price Tier' column as well. To see if I will be able to find a relationship between 'Likes' and 'Price Tier', and thus fill the missing values with a prediction, I will make a scatter plot between the two columns. See below for the plot:

![likes_against_price](images/likes_against_price)

As seen from the above, there is no clear relationship between the two columns. Moreover, since I am only seeking to predict a restaurant's popularity based on locaton, the 'Price Tier' variable should not be part of the feature set. Thus, I will drop the 'Price Tier' column.

Now, I will have to retrieve all the nearby venues for each chinese restaurant and onehot encode them so that I will be able to fit the data into various Machine Learning models. To retrieve venues that are within 1km of each chinese restaurant, I will be using the [explore method](https://developer.foursquare.com/docs/api-reference/venues/explore/).

The resulting dataframe is as follows:

![onehot_likes](images/onehot_likes)

Since I will be training the models to classify them into 3 categories - Popular, Average, Unpopular - I have decided to take the top 25 percentile most liked restaurant and labeled them as 'Popular', while the bottom 25 percentile liked restaurants are labeled as 'Unpopular'. The remaining restaurants have been labeled 'Average' under the 'Popularity' column.

See the below dataframe:

![onehot_popularity](images/onehot_popularity)

### 3.3 Feature Selection

From the onehot encoded dataframe, I removed the columns 'Neighborhood', 'Restaurant ID', 'Restaurant Name', 'Restaurant Latitude', 'Restaurant Longitude', 'Popularity', 'Price Tier', 'Likes', keeping only the existence of a venue in the viscinity of the chinese restaurant. See below.

![features](images/features)

The column 'Popularity' was assigned to the outcome vector variable y.

## References
* [Migration Policy Stats](https://www.migrationpolicy.org/article/chinese-immigrants-united-states)
* [New York Neighborhoods](https://ibm.box.com/shared/static/fbpwbovar7lf8p5sgddm06cgipa2rxpe.json)
* [Foursquare API](https://developer.foursquare.com/docs/places-api/)

