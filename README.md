# MarketGAN
#### Implementing a Generative Adversarial Network (GAN) on the stock market through a pipeline on Google Colab. Data used from 500 Companies from S&P500, downloaded by Alpha Vantage, and trained using a 3-Layer Dense Network as the Generator and a 3-Layer Convolutional Neural Network as the Discriminator.

#### Sources: 

Modified and reused code from https://github.com/MiloMallo/StockMarketGAN which was sourced from https://github.com/nmharmon8/StockMarketGAN. 

#### [An Example Colab Notebook](https://github.com/kah-ve/MarketGAN/blob/master/stock-market-gan-11-29-s-p-500-50k-epochs-20-history-5-days-ahead-1-pct_change.ipynb) 

#### Abstract

Neural networks have been advancing in capability very rapidly in recent years. One of the newest techniques with these networks is Generative Adversarial Networks. In this GAN architecture you have two neural networks pitted against each other, one trying to fool the other with noise, while the other trains on real data and responds with information on how to make that noise more realistic. After many runs, you would ideally be able to generate data that the other network wouldn't know was real or fake. We aim to implement this powerful method in the modeling of time series data, with our current medium being the stock market. A GAN that is able to work well with time series data, especially chaotic ones such as the market, would be very useful in many other areas. One is finance where you can better predict the risk in an investment, but another application might be in effectively anonymizing private sensitive data. A lot of data today is not shared because of confidentiality, so being able to generate accurate synthetic versions would allow one to release the data to the public without worry.

#### Setup

Built a pipeline on Google Colab (offers a free K80 GPU for 12-hour sessions). Can be tedious to setup but works like a charm after since we could access it anywhere, change some parameters, and train a model. Some variables we played around with are different lists of companies, number of epochs, days to predict with, days to predict ahead, and threshold of percentage change to predict on. We built on top of code found on Github and added many modifications, some of which are adding methods to stop training and view confusion matrices, streamlining the process to deploy files for predictions, adapting the code to work with Google Colab, and allowing for prompt parameter changes.

#### Results

Using 30 company stocks based on highest market cap as we initially planned turned out to be completely unhelpful. Moved on to S&P 500 because it accounted for 80% of movement in the market. Shown below are the GAN results from the S&P 500 companies, 20 historical days, 5 days ahead predictions, 50k epochs, and various levels of percentage threshold. For predictions of simply up or down (0% threshold), we see that the GAN has decent results, though a CNN we also trained alongside it (not shown below) still came out slightly better. In terms of predicting a 10% change, the GAN does quite badly. It seems to predict most of the down movements correctly but almost none of the up. Loosening the threshold to 1%, we see that there is actually a significant change in up predictions compared to the 10% threshold. We are now predicting 14% of the true ups rather than just 1% of them, while losing very little of the accuracy in down predictions.

![Results](https://github.com/kah-ve/MarketGAN/blob/master/GANResults.PNG) 

#### Future Work

We've barely scratched the surface with what we can do with GANs, for we have only set up a framework and a complete data pipeline. There can be a lot of improvements in terms of the type of layers and the depth of the layers we are using. This requires us to look into more research paper on this subject. Moreover, by adding more performance measures’ metrics, we will be able to tune different parameters, add more relevant indicators in our training, try a different selection of companies, and even improve the pipeline we have built so far. One of the main goals for next semester the goal is to build the GANs using recurrent neural networks.

