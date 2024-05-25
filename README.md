# Predicting Breast Cancer Using Machine Learning

![predicting-breast-cancer-using-machine-learning.png](images/predicting-breast-cancer-using-machine-learning.png)

Imagine harnessing the power of machine learning to predict one of the most prevalent and life-threatening diseases: breast cancer. As data science enthusiasts, we often seek new challenges to expand our skills and dive into unexplored territories. This journey not only enhances our technical prowess but also broadens our understanding of diverse fields.

This article invites you to venture beyond the realms of
`digital marketing` and `media investment` into the captivating world of `healthcare`. Did you know that cancer is the second leading cause of death globally, accounting for approximately 9.6 million deaths in 2018,
according to the [WHO](https://www.who.int/news-room/fact-sheets/detailcancer#:~:text=Cancer%20is%20the%20second%20leading,-%20and%20middle-income%20countries.). This staggering statistic underscores the urgent need for innovative solutions in early detection and treatment.

Join me as we explore how machine learning can be a game-changer in predicting breast cancer symptoms. Well utilize a comprehensive dataset from [UCI](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29), generously provided by academicians, to build our predictive model.

To bring this vision to life, well employ powerful Python libraries like [Pandas](https://pandas.pydata.org/), [Seaborn](https://seaborn.pydata.org/), and [Scikit-learn](https://scikit-learn.org/). These tools will help us explore, clean, and visualize data, ultimately leading to a robust machine learning model. Ready to embark on this exciting adventure? Lets break it down into manageable steps:

1.  **Loading Libraries**
2.  **Data Exploration**
3.  **Data Visualization**
4.  **One Hot Encoding**
5.  **Feature Generation**
6.  **Data Splitting**
7.  **Machine Learning Modeling**
8.  **Data Prediction**

Dive in and discover how you can leverage machine learning to make a meaningful impact in the fight against breast cancer.

## Load Libraries

Much like any other data exploratory process in Pandas or Python, the initial phase involves loading the essential libraries into our working Jupyter Notebook environment. These libraries are the backbone of our data analysis and machine learning endeavors, providing us with the tools needed to manipulate, visualize, and model our data. Whether
youre using Jupyter Notebook, Google Colab, or Kaggle, the process remains largely the same. These platforms offer robust environments that support Python and its libraries, making them ideal for data science projects.

For this tutorial, I'll stick to my faithful Jupyter Notebook environment, known for its versatility and user-friendly interface. Jupyter Notebook allows for an interactive data analysis experience, where code, visualizations, and explanatory text can coexist seamlessly. This setup will enable us to document our process comprehensively and adjust our code on the fly as we delve into the breast cancer dataset. While youre free to use any Integrated Development Environment (IDE) you prefer, Jupyter Notebooks integration with libraries like Pandas, Seaborn, and Scikit-learn makes it an excellent choice for this step-by-step guide.

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
```

## Load Dataset

Start by creating a directory on your computer. Although Im using a MacOS environment, the instructions provided here are applicable across different platforms. For the purpose of this walkthrough, lets name the directory `Project`. This will serve as our main working directory. Navigate into the `Project` folder, as this will be our base for organizing and executing the steps outlined in this tutorial. The next step is to download the breast cancer dataset from the [UCI](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29) site, which well use for our machine learning model.

Within the `Project` directory, create a new folder named `data` and copy the downloaded CSV file into this `data` directory. This organization ensures that all relevant files are neatly stored and easily accessible throughout the tutorial. By structuring our project this way, we facilitate a smooth workflow and maintain order as we progress. Now that everything is set up, we can load the dataset into our Jupyter Notebook. This step allows us to examine, manipulate, and observe the data, laying the groundwork for our machine learning exploration.

```python
df = pd.read_csv('data/breast_cancer_data.csv')
```

## Dataset Size

Once we have completed the initial setup, we can proceed to analyze our dataset further. A common starting point in any data analysis project is to understand the size of the dataset. You might be wondering, just how large is our dataset? This question is easily answered using the `.shape` method in Pandas.

By applying the `.shape` method to our dataset, we can quickly obtain the number of rows and columns. This method returns a tuple representing the dimensions of the dataset, giving us an immediate sense of its scale. Understanding the size of our dataset is crucial as it informs us about the volume of data we will be working with and helps in planning subsequent data processing and analysis steps.

```python
df.shape
```

### Rows & Columns

We can see we have the following information at hand:

- rows `699`
- columns `12`

## Data Types

Its always a good idea to get cozy with our dataset, not just by looking at its size, but by understanding what its really made of. Think of it like getting to know a new friend---you wouldn't just ask them how tall they are, right? Youd want to know their quirks, their traits, what makes them tick. The same goes for our data. Knowing the types of data in each column helps us groove through the feature generation phase with ease.

So, let's kick back and take a deeper dive. By checking out the data types of each column, we get the full picture: the numbers, the categories, the text. This insight is like the smooth rhythm of a jazz tune, guiding us to apply the right transformations and manipulations. When were in sync with our data, everything just flows better, leading to more accurate and reliable models. To get this vibe going, well use the `.dtypes` attribute in Pandas. It's our backstage pass to the inner workings of the dataset, giving us a clear overview of the structure and content. Lets get jazzy with our data and see what it's composed of!

```python
# We need to observe the data types of each columns
df.dtypes
```

- patient_id int64
- clump_thickness float64
- cell_size_uniformity float64
- cell_shape_uniformity int64
- marginal_adhesion int64
- single_ep_cell_size int64
- bare_nuclei object
- bland_chromatin float64
- normal_nucleoli float64
- mitoses int64
- class object
- doctor_name object
  dtype: object

### The Data Legend

Lets lay down the smooth beats of our dataset. Heres the lowdown on the columns we have, as described by the source:

- `Patient ID`: id number
- `Clump Thickness`: 1--10
- `Uniformity of Cell Size`: 1--10
- `Uniformity of Cell Shape`: 1--10
- `Marginal Adhesion`: 1--10
- `Single Epithelial Cell Size`: 1--10
- `Bare Nuclei`: 1--10
- `Bland Chromatin`: 1--10
- `Normal Nucleoli`: 1--10
- `Mitoses`: 1--10
- `Class`: malignant or benign
- `Doctor name`: 4 different doctors

So, what's the vibe here? The `Patient ID` is our unique identifier, ensuring each record stands out. The `Class` column is the headline act, telling us whether the tumor is `malignant` (cancerous) or `benign` (not cancerous). The rest of the columns? Theyre numeric medical descriptions of the tumor, except for `Doctor name`, which adds a categorical twist.

> Keep this in mind---if our goal is to predict whether a tumor is
> cancerous based on the other features, we'll need to perform some
> one-hot encoding on the categorical data and clean up the numerical
> data. Just like tuning an instrument before a jam session, prepping
> our data ensures everything flows smoothly in our analysis.

## First & Last Rows

Now that we've got the lay of the land, lets dive in and see what the top five records in our dataset look like. This peek at the first few rows will give us a quick feel for the data and help us spot any obvious issues or patterns right off the bat.

To do this, well use the `.head()` method in Pandas, which will show us the first five rows. Its like getting a sneak preview of the opening act before the main event. This simple step is crucial for ensuring were on the right track and that our data is ready to roll.

```python
df.head()
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
				<th>class</th>
				<th>doctor_name</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>0</th>
				<td>1000025</td>
				<td>5.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Doe</td>
			</tr>
			<tr>
				<th>1</th>
				<td>1002945</td>
				<td>5.0</td>
				<td>4.0</td>
				<td>4</td>
				<td>5</td>
				<td>7</td>
				<td>10</td>
				<td>3.0</td>
				<td>2.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>2</th>
				<td>1015425</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>2</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>3</th>
				<td>1016277</td>
				<td>6.0</td>
				<td>8.0</td>
				<td>8</td>
				<td>1</td>
				<td>3</td>
				<td>4</td>
				<td>3.0</td>
				<td>7.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>4</th>
				<td>1017023</td>
				<td>4.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>3</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Wong</td>
			</tr>
		</tbody>
	</table>
</div>
```

Additionally, checking the last few records with the `.tail()` method will give us a complete sense of the datasets structure. This combination of the first and last rows provides a balanced overview, ensuring no surprises lurk at the end. Lets groove through the data and see what stories the top and bottom rows tell us!

```python
df.tail()
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
				<th>class</th>
				<th>doctor_name</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>694</th>
				<td>776715</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>3</td>
				<td>2</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>695</th>
				<td>841769</td>
				<td>2.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>696</th>
				<td>888820</td>
				<td>5.0</td>
				<td>10.0</td>
				<td>10</td>
				<td>3</td>
				<td>7</td>
				<td>3</td>
				<td>8.0</td>
				<td>10.0</td>
				<td>2</td>
				<td>malignant</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>697</th>
				<td>897471</td>
				<td>4.0</td>
				<td>8.0</td>
				<td>6</td>
				<td>4</td>
				<td>3</td>
				<td>4</td>
				<td>10.0</td>
				<td>6.0</td>
				<td>1</td>
				<td>malignant</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>698</th>
				<td>897471</td>
				<td>4.0</td>
				<td>8.0</td>
				<td>8</td>
				<td>5</td>
				<td>4</td>
				<td>5</td>
				<td>10.0</td>
				<td>4.0</td>
				<td>1</td>
				<td>malignant</td>
				<td>Dr. Wong</td>
			</tr>
		</tbody>
	</table>
</div>
```

## Descriptive Statistics

### Numerical Analysis

Lets jazz up our dataset with some sweet statistical insights! With the `.describe()` method, were about to dive deep into the numerical nitty-gritty. This little trick gives us the lowdown on key stats like `count`, `mean`, and `standard deviation`, shedding light on the distribution and central tendencies of our numeric data.

So, why does this matter? Well, getting cozy with these numbers gives us a clearer picture of what we're working with. Its like fine-tuning our instruments before a performance---it ensures our analysis hits all the right notes. With these stats in hand, we can groove through our dataset with confidence, uncovering hidden patterns and trends along the way. Lets crank up the volume and see what the numbers have to say! 🎶

```python
df.describe()
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>count</th>
				<td>6.990000e+02</td>
				<td>698.000000</td>
				<td>698.000000</td>
				<td>699.000000</td>
				<td>699.000000</td>
				<td>699.000000</td>
				<td>695.000000</td>
				<td>698.000000</td>
				<td>699.000000</td>
			</tr>
			<tr>
				<th>mean</th>
				<td>1.071704e+06</td>
				<td>4.416905</td>
				<td>3.137536</td>
				<td>3.207439</td>
				<td>2.793991</td>
				<td>3.216023</td>
				<td>3.447482</td>
				<td>2.868195</td>
				<td>1.589413</td>
			</tr>
			<tr>
				<th>std</th>
				<td>6.170957e+05</td>
				<td>2.817673</td>
				<td>3.052575</td>
				<td>2.971913</td>
				<td>2.843163</td>
				<td>2.214300</td>
				<td>2.441191</td>
				<td>3.055647</td>
				<td>1.715078</td>
			</tr>
			<tr>
				<th>min</th>
				<td>6.163400e+04</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
			</tr>
			<tr>
				<th>25%</th>
				<td>8.706885e+05</td>
				<td>2.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>2.000000</td>
				<td>2.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
			</tr>
			<tr>
				<th>50%</th>
				<td>1.171710e+06</td>
				<td>4.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
				<td>2.000000</td>
				<td>3.000000</td>
				<td>1.000000</td>
				<td>1.000000</td>
			</tr>
			<tr>
				<th>75%</th>
				<td>1.238298e+06</td>
				<td>6.000000</td>
				<td>5.000000</td>
				<td>5.000000</td>
				<td>3.500000</td>
				<td>4.000000</td>
				<td>5.000000</td>
				<td>4.000000</td>
				<td>1.000000</td>
			</tr>
			<tr>
				<th>max</th>
				<td>1.345435e+07</td>
				<td>10.000000</td>
				<td>10.000000</td>
				<td>10.000000</td>
				<td>10.000000</td>
				<td>10.000000</td>
				<td>10.000000</td>
				<td>10.000000</td>
				<td>10.000000</td>
			</tr>
		</tbody>
	</table>
</div>
```

## Categorical Analysis

Just like tuning into a different frequency, lets shift our focus to the categorical side of the spectrum. With the `.describe(include=['O'])` method, were about to unravel the mysteries of our categorical data. While the output might be a bit more concise compared to its numerical counterpart, it still packs a punch.

By honing in on the categorical variables---those with a data type of `object`---we gain valuable insights into their distribution and uniqueness. Its like flipping through the pages of a well-worn record collection, each category offering its own distinct vibe.

So, why bother? Well, understanding the landscape of our categorical data sets the stage for deeper analysis. Just like a DJ crafting the perfect mix, these insights help us blend and remix our data with precision. With the `.describe(include=['O'])` method in hand, were ready to spin some categorical magic and uncover the stories hidden within our dataset. Lets dive in and see what melodies await! 🎵

```python
df.describe(include=['O'])
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>bare_nuclei</th>
				<th>class</th>
				<th>doctor_name</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>count</th>
				<td>697</td>
				<td>699</td>
				<td>699</td>
			</tr>
			<tr>
				<th>unique</th>
				<td>11</td>
				<td>2</td>
				<td>4</td>
			</tr>
			<tr>
				<th>top</th>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Doe</td>
			</tr>
			<tr>
				<th>freq</th>
				<td>401</td>
				<td>458</td>
				<td>185</td>
			</tr>
		</tbody>
	</table>
</div>
```

## Data Reshaping

Time to remix our data and give it a fresh new vibe! With the code snippet youve got in hand, were about to shake things up and reshape our dataset like never before. By grooving to the beat of `df.groupby(by=['doctor_name', 'class']).count()`, were taking our data on a whole new journey.

Picture this: were gathering our data around the DJ booth, grouping it by the soothing sounds of `doctor_name` and the electrifying beats of `class`. Then, we crank up the volume with the aggregation function, counting up the hits in each group. Its like taking our dataset to a cool underground club, where every combination of doctor and class brings its own unique vibe.

Why does this matter? Well, reshaping our data in this way allows us to uncover fresh insights and patterns that might have been hidden before. Its like remixing a classic track---same ingredients, but with a whole new flavor. So, grab your data and lets hit the dance floor, because were about to reshape it into something truly groovy! 🎧💃

```python
# This aggreates the data by its column names, then we pass the aggregation function (size = count)
df.groupby(by =['doctor_name', 'class']).count()
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
			</tr>
			<tr>
				<th>doctor_name</th>
				<th>class</th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th
					rowspan="2"
					valign="top">
					Dr. Doe
				</th>
				<th>benign</th>
				<td>127</td>
				<td>127</td>
				<td>126</td>
				<td>127</td>
				<td>127</td>
				<td>127</td>
				<td>126</td>
				<td>126</td>
				<td>127</td>
				<td>127</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>57</td>
				<td>58</td>
				<td>58</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					Dr. Lee
				</th>
				<th>benign</th>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>119</td>
				<td>121</td>
				<td>121</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					Dr. Smith
				</th>
				<th>benign</th>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>74</td>
				<td>73</td>
				<td>74</td>
				<td>74</td>
				<td>74</td>
				<td>74</td>
				<td>73</td>
				<td>74</td>
				<td>74</td>
				<td>74</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					Dr. Wong
				</th>
				<th>benign</th>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>107</td>
				<td>108</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
			</tr>
		</tbody>
	</table>
</div>
```

```python
df.groupby(by =['class', 'doctor_name']).count()
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
			</tr>
			<tr>
				<th>class</th>
				<th>doctor_name</th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th
					rowspan="4"
					valign="top">
					benign
				</th>
				<th>Dr. Doe</th>
				<td>127</td>
				<td>127</td>
				<td>126</td>
				<td>127</td>
				<td>127</td>
				<td>127</td>
				<td>126</td>
				<td>126</td>
				<td>127</td>
				<td>127</td>
			</tr>
			<tr>
				<th>Dr. Lee</th>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>121</td>
				<td>119</td>
				<td>121</td>
				<td>121</td>
			</tr>
			<tr>
				<th>Dr. Smith</th>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
				<td>102</td>
			</tr>
			<tr>
				<th>Dr. Wong</th>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>108</td>
				<td>107</td>
				<td>108</td>
			</tr>
			<tr>
				<th
					rowspan="4"
					valign="top">
					malignant
				</th>
				<th>Dr. Doe</th>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>58</td>
				<td>57</td>
				<td>58</td>
				<td>58</td>
			</tr>
			<tr>
				<th>Dr. Lee</th>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
				<td>60</td>
			</tr>
			<tr>
				<th>Dr. Smith</th>
				<td>74</td>
				<td>73</td>
				<td>74</td>
				<td>74</td>
				<td>74</td>
				<td>74</td>
				<td>73</td>
				<td>74</td>
				<td>74</td>
				<td>74</td>
			</tr>
			<tr>
				<th>Dr. Wong</th>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
				<td>49</td>
			</tr>
		</tbody>
	</table>
</div>
```

```python
df.groupby(by =['bare_nuclei', 'class']).count()
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
				<th>doctor_name</th>
			</tr>
			<tr>
				<th>bare_nuclei</th>
				<th>class</th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
				<th></th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th
					rowspan="2"
					valign="top">
					1
				</th>
				<th>benign</th>
				<td>386</td>
				<td>386</td>
				<td>386</td>
				<td>386</td>
				<td>386</td>
				<td>386</td>
				<td>383</td>
				<td>385</td>
				<td>386</td>
				<td>386</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>15</td>
				<td>15</td>
				<td>15</td>
				<td>15</td>
				<td>15</td>
				<td>15</td>
				<td>15</td>
				<td>15</td>
				<td>15</td>
				<td>15</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					10
				</th>
				<th>benign</th>
				<td>3</td>
				<td>3</td>
				<td>2</td>
				<td>3</td>
				<td>3</td>
				<td>3</td>
				<td>3</td>
				<td>3</td>
				<td>3</td>
				<td>3</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>128</td>
				<td>128</td>
				<td>128</td>
				<td>128</td>
				<td>128</td>
				<td>128</td>
				<td>128</td>
				<td>128</td>
				<td>128</td>
				<td>128</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					2
				</th>
				<th>benign</th>
				<td>21</td>
				<td>21</td>
				<td>21</td>
				<td>21</td>
				<td>21</td>
				<td>21</td>
				<td>21</td>
				<td>21</td>
				<td>21</td>
				<td>21</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					3
				</th>
				<th>benign</th>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>14</td>
				<td>13</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					4
				</th>
				<th>benign</th>
				<td>6</td>
				<td>6</td>
				<td>6</td>
				<td>6</td>
				<td>6</td>
				<td>6</td>
				<td>6</td>
				<td>6</td>
				<td>6</td>
				<td>6</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>13</td>
				<td>13</td>
				<td>13</td>
				<td>13</td>
				<td>13</td>
				<td>13</td>
				<td>13</td>
				<td>13</td>
				<td>13</td>
				<td>13</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					5
				</th>
				<th>benign</th>
				<td>10</td>
				<td>10</td>
				<td>10</td>
				<td>10</td>
				<td>10</td>
				<td>10</td>
				<td>10</td>
				<td>10</td>
				<td>10</td>
				<td>10</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>20</td>
				<td>20</td>
				<td>20</td>
				<td>20</td>
				<td>20</td>
				<td>20</td>
				<td>20</td>
				<td>20</td>
				<td>20</td>
				<td>20</td>
			</tr>
			<tr>
				<th>6</th>
				<th>malignant</th>
				<td>4</td>
				<td>4</td>
				<td>4</td>
				<td>4</td>
				<td>4</td>
				<td>4</td>
				<td>4</td>
				<td>4</td>
				<td>4</td>
				<td>4</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					7
				</th>
				<th>benign</th>
				<td>1</td>
				<td>1</td>
				<td>1</td>
				<td>1</td>
				<td>1</td>
				<td>1</td>
				<td>1</td>
				<td>1</td>
				<td>1</td>
				<td>1</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>7</td>
				<td>7</td>
				<td>7</td>
				<td>7</td>
				<td>7</td>
				<td>7</td>
				<td>6</td>
				<td>7</td>
				<td>7</td>
				<td>7</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					8
				</th>
				<th>benign</th>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>19</td>
				<td>19</td>
				<td>19</td>
				<td>19</td>
				<td>19</td>
				<td>19</td>
				<td>19</td>
				<td>19</td>
				<td>19</td>
				<td>19</td>
			</tr>
			<tr>
				<th>9</th>
				<th>malignant</th>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
				<td>9</td>
			</tr>
			<tr>
				<th
					rowspan="2"
					valign="top">
					?
				</th>
				<th>benign</th>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
				<td>14</td>
			</tr>
			<tr>
				<th>malignant</th>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
				<td>2</td>
			</tr>
		</tbody>
	</table>
</div>
```

# Data Cleaning

Alright, time to chill and tidy up our dataset! Now that we've wrapped up the early analysis phase, its onto the next groove: cleaning up our data. Picture this: your data rolls in with all sorts of shapes and sizes, like records in a crate waiting to be sorted. But the real magic happens when we polish it up, turning it into the complete and comprehensive masterpiece we need.

Sure, its like sifting through a crate of vinyl, each record with its own scratches and dust. But trust me, the best jams come from the cleanest cuts. By whipping our dataset into shape, were setting the stage for some serious feature engineering and analysis down the line. So, grab your data mop and broom, because were about to sweep away the dust and uncover the smooth grooves beneath. Lets get cleaning! 🎶✨

## Missing Records

Among one of the easiet way to identify whether or not your dataset has any missing data in them, would be to check them using the `.isna()` method and combine them with the `.sum()` function. It would in return, give you information on how many rows gone missing in your current dataset. Usually Pandas, would assing them with the value of `NaN`, but it can always be just a blank value in the record cell.

```python
df.isna().sum()
```

- patient_id 0
- clump_thickness 1
- cell_size_uniformity 1
- cell_shape_uniformity 0
- marginal_adhesion 0
- single_ep_cell_size 0
- bare_nuclei 2
- bland_chromatin 4
- normal_nucleoli 1
- mitoses 0
- class 0
- doctor_name 0
  dtype: int64

Good to know that the `patient_id` has 0 missing values, but as you may notice, others columns much like `clump_thickness`,`cell_size_uniformity`, `bare_nuclei`, `bland_chromatin` and`normal_nucleoli`, and to put them in total, there are 9 missing rows in the dataset.

## How To Deal With?

The real question isnt just about spotting the missing records and summing them up. The real jazz starts when you decide how to handle them before moving forward on your data wrangling journey. In this particular case, weve got a small amount of data with missing values---just `9 rows` out of `699`. Thats a mere `0.012`, or less than 1% of the total dataset. With such a small fraction, Im thinking we drop them like they're hot, using the `.dropna` method. And while were at it, let's break down the cool attributes that groove along with the `.dropna` method.

- `Axis`: Decides if youre dropping rows or columns. `0` means rows,
  while `1` goes for columns.
- `How`: Two vibes here---_any_ or _all_. If you choose `all`, it
  drops rows or columns that are completely empty. Opt for `any`, and
  it drops those with even a single missing value.
- `Inplace`: This ones crucial. If you set `inplace=True`, changes
  will happen right on the DataFrame youre working with. If its
  `False` (which is the default), the original DataFrame stays
  untouched, and a new one is returned.

So, lets clean up those missing beats and keep the data flowing
smoothly!

```python
df.dropna(axis=0, how='any', inplace=True)
```

```python
df
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
				<th>class</th>
				<th>doctor_name</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>0</th>
				<td>1000025</td>
				<td>5.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Doe</td>
			</tr>
			<tr>
				<th>1</th>
				<td>1002945</td>
				<td>5.0</td>
				<td>4.0</td>
				<td>4</td>
				<td>5</td>
				<td>7</td>
				<td>10</td>
				<td>3.0</td>
				<td>2.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>2</th>
				<td>1015425</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>2</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>3</th>
				<td>1016277</td>
				<td>6.0</td>
				<td>8.0</td>
				<td>8</td>
				<td>1</td>
				<td>3</td>
				<td>4</td>
				<td>3.0</td>
				<td>7.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>4</th>
				<td>1017023</td>
				<td>4.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>3</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Wong</td>
			</tr>
			<tr>
				<th>...</th>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
			</tr>
			<tr>
				<th>694</th>
				<td>776715</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>3</td>
				<td>2</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>695</th>
				<td>841769</td>
				<td>2.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>696</th>
				<td>888820</td>
				<td>5.0</td>
				<td>10.0</td>
				<td>10</td>
				<td>3</td>
				<td>7</td>
				<td>3</td>
				<td>8.0</td>
				<td>10.0</td>
				<td>2</td>
				<td>malignant</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>697</th>
				<td>897471</td>
				<td>4.0</td>
				<td>8.0</td>
				<td>6</td>
				<td>4</td>
				<td>3</td>
				<td>4</td>
				<td>10.0</td>
				<td>6.0</td>
				<td>1</td>
				<td>malignant</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>698</th>
				<td>897471</td>
				<td>4.0</td>
				<td>8.0</td>
				<td>8</td>
				<td>5</td>
				<td>4</td>
				<td>5</td>
				<td>10.0</td>
				<td>4.0</td>
				<td>1</td>
				<td>malignant</td>
				<td>Dr. Wong</td>
			</tr>
		</tbody>
	</table>
	<p>690 rows × 12 columns</p>
</div>
```

As you can see now, the rows number have been decreased, from `699` to `690`, down with `9` records, but left us with clean dataset with no empty cells in them. Let move on to check them!

```python
df.isnull().values.any()
```

False

## Rechecking

Now that we've got our dataset shining bright with no empty records, we might still be wondering if there's another way to double-check for any sneaky missing values. Good news, data groovers! Theres a slick method called `.isnull` that performs a boolean check, giving you a smooth true or false response to your inquiry. Its like having a jazz soloist confirming every note is in place. So, lets slide into it and make sure our dataset is as clean as a crisp vinyl record. Let's do this!
🎷✨

## Validating

So, our dataset's looking sharp, but let's not stop there. If you're curious whether there are still any hidden empty cells lurking around, theres a cool cat method called `.isnull` that's perfect for the job. This boolean checker will let us know with a simple true or false if any values are missing. It's like having an extra pair of ears in the studio, ensuring every beat is perfect. Let's give it a spin and make sure everything's in tip-top shape! 🎶🔍

```python
df.isnull()
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
				<th>class</th>
				<th>doctor_name</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>0</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>1</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>2</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>3</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>4</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>...</th>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
			</tr>
			<tr>
				<th>694</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>695</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>696</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>697</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
			<tr>
				<th>698</th>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
				<td>False</td>
			</tr>
		</tbody>
	</table>
	<p>690 rows × 12 columns</p>
</div>
```

So far, so groovy! From our dataset checks, everything's coming back with `False` values, and thats music to our ears. It means one thing: our dataset is spotless and ready to jam.

With our data all tuned up, its time to move on to the next leg of our journey. So, lets keep the rhythm going and dive into the next adventure. Onward to data greatness! 🎷✨

## Duplicate Records

Alright, lets jazz things up and hunt for those duplicate records! First, we need to investigate whether our dataset is hiding any duplicate grooves in the cell records. By getting on top of this early, we can dodge potential hurdles that might throw our analysis offbeat and introduce unwanted bias.

To kick off this detective work, we'll use the `.nunique` method. This little gem will give us some solid pointers to detect any anomalies lurking in our dataset. We'll start by grooving through the columns that are supposed to have unique identifiers---those special `object` datatype columns. In our case, it's the `patient_id` column. So, let's spin that record and see if we have any duplicates in the mix! 🎷🔍

```python
df.nunique()
```

- patient_id 637
- clump_thickness 10
- cell_size_uniformity 10
- cell_shape_uniformity 10
- marginal_adhesion 10
- single_ep_cell_size 10
- bare_nuclei 11
- bland_chromatin 10
- normal_nucleoli 10
- mitoses 9
- class 2
- doctor_name 4
  dtype: int64

```python
df.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
Index: 690 entries, 0 to 698
Data columns (total 12 columns): # Column Non-Null Count Dtype
```

---

- 0 patient_id 690 non-null int64
- 1 clump_thickness 690 non-null float64
- 2 cell_size_uniformity 690 non-null float64
- 3 cell_shape_uniformity 690 non-null int64
- 4 marginal_adhesion 690 non-null int64
- 5 single_ep_cell_size 690 non-null int64
- 6 bare_nuclei 690 non-null object
- 7 bland_chromatin 690 non-null float64
- 8 normal_nucleoli 690 non-null float64
- 9 mitoses 690 non-null int64
- 10 class 690 non-null object
- 11 doctor_name 690 non-null object
  dtypes: float64(4), int64(5), object(3)
  memory usage: 70.1+ KB

We know for sure that our dataset is jamming with `690` rows and `12` columns (down from the previous `699`). But hold on a second---when we dig into the groove, we find that `patient_id` only has `637` records. Somethings offbeat here, especially since `patient_id` should be our unique identifier. We should be seeing a solid `690` records, not just`637`.

Time to put on our detective hats and investigate this mystery. There's gotta be some duplication in the `patient_id` column messing with our flow. Let's dive deep, spin those records backwards, and uncover where the duplicates are hiding. This dataset is about to get a clean remix! 🎷🔍

### Duplicate Patients

Alright, buckle up, because we're about to dive into the mystery of the duplicate `patient_id` records. Picture this: you're flipping through your dataset like a detective, and suddenly, you stumble upon some suspicious duplicates. But fear not, because weve got just the solution to unravel this enigma.

Were borrowing a slick move from the data science playbook, courtesy of the wizards over at Stack Overflow. This little trick is like shining a spotlight on the shadows, revealing all the duplicate items lurking in the shadows of our dataset. With this solution in hand, well shine a light on those repeat offenders and get to the bottom of this duplication dilemma. So, get ready to crack the case and uncover the truth behind those duplicate patients! 🕵️‍♂️🔍

- Borrowed from
  <https://stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all-the-duplicate-items-using-pandas-in-python>

```python
df[df.patient_id.duplicated(keep=False)].sort_values("patient_id")
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
				<th>class</th>
				<th>doctor_name</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>267</th>
				<td>320675</td>
				<td>3.0</td>
				<td>3.0</td>
				<td>5</td>
				<td>2</td>
				<td>3</td>
				<td>10</td>
				<td>7.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>malignant</td>
				<td>Dr. Wong</td>
			</tr>
			<tr>
				<th>272</th>
				<td>320675</td>
				<td>3.0</td>
				<td>3.0</td>
				<td>5</td>
				<td>2</td>
				<td>3</td>
				<td>10</td>
				<td>7.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>malignant</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>575</th>
				<td>385103</td>
				<td>5.0</td>
				<td>1.0</td>
				<td>2</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>269</th>
				<td>385103</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Doe</td>
			</tr>
			<tr>
				<th>271</th>
				<td>411453</td>
				<td>5.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Wong</td>
			</tr>
			<tr>
				<th>...</th>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
			</tr>
			<tr>
				<th>560</th>
				<td>1321942</td>
				<td>5.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Doe</td>
			</tr>
			<tr>
				<th>660</th>
				<td>1339781</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>2.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>661</th>
				<td>1339781</td>
				<td>4.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>672</th>
				<td>1354840</td>
				<td>2.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Wong</td>
			</tr>
			<tr>
				<th>673</th>
				<td>1354840</td>
				<td>5.0</td>
				<td>3.0</td>
				<td>2</td>
				<td>1</td>
				<td>3</td>
				<td>1</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
		</tbody>
	</table>
	<p>98 rows × 12 columns</p>
</div>
```

Alright, check this out: weve got a little situation on our hands. It seems like weve got `98` patient IDs making multiple appearances in our dataset. Some are showing up twice, while others are pulling off the triple play. Now, wouldn't it be sweet if we could get the lowdown on exactly how many times each patient ID is making a cameo?

Well, guess what? Were about to dive into the nitty-gritty and unravel this mystery. Picture this: were peeling back the layers of duplication, analyzing each instance to tally up the total count. Its like detective work for data scientists---sleuthing through the numbers to uncover the truth. So, grab your magnifying glass and lets crack this case wide open. Were diving deep into the world of duplications, ready to count em up and bring clarity to our dataset! 🕵️‍♂️🔍

### Count Duplications

Lets analyze how many times a single `patient_id` value, was being recorded more than once, in the next table.

- Borrowed from
  <https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas-per-groups>

```python
df.patient_id.value_counts()
```

patient_id

- 1182404 6
- 1276091 5
- 1198641 3
- 897471 2
- 411453 2
  ..
- 1231706 1
- 1232225 1
- 1236043 1
- 1241232 1
- 809912 1
  Name: count, Length: 637, dtype: int64

Surprisingly, some are getting recorded more than twice, some are even getting recorded 6 times. Lets move on to the next steps on how to deal with them.

```python
df.drop_duplicates(subset="patient_id", keep='first', inplace = True)
df # let's print them.
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
				<th>class</th>
				<th>doctor_name</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>0</th>
				<td>1000025</td>
				<td>5.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Doe</td>
			</tr>
			<tr>
				<th>1</th>
				<td>1002945</td>
				<td>5.0</td>
				<td>4.0</td>
				<td>4</td>
				<td>5</td>
				<td>7</td>
				<td>10</td>
				<td>3.0</td>
				<td>2.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>2</th>
				<td>1015425</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>2</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>3</th>
				<td>1016277</td>
				<td>6.0</td>
				<td>8.0</td>
				<td>8</td>
				<td>1</td>
				<td>3</td>
				<td>4</td>
				<td>3.0</td>
				<td>7.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>4</th>
				<td>1017023</td>
				<td>4.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>3</td>
				<td>2</td>
				<td>1</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Wong</td>
			</tr>
			<tr>
				<th>...</th>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
				<td>...</td>
			</tr>
			<tr>
				<th>693</th>
				<td>763235</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>2.0</td>
				<td>1.0</td>
				<td>2</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>694</th>
				<td>776715</td>
				<td>3.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>3</td>
				<td>2</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>695</th>
				<td>841769</td>
				<td>2.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>1.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Smith</td>
			</tr>
			<tr>
				<th>696</th>
				<td>888820</td>
				<td>5.0</td>
				<td>10.0</td>
				<td>10</td>
				<td>3</td>
				<td>7</td>
				<td>3</td>
				<td>8.0</td>
				<td>10.0</td>
				<td>2</td>
				<td>malignant</td>
				<td>Dr. Lee</td>
			</tr>
			<tr>
				<th>697</th>
				<td>897471</td>
				<td>4.0</td>
				<td>8.0</td>
				<td>6</td>
				<td>4</td>
				<td>3</td>
				<td>4</td>
				<td>10.0</td>
				<td>6.0</td>
				<td>1</td>
				<td>malignant</td>
				<td>Dr. Lee</td>
			</tr>
		</tbody>
	</table>
	<p>637 rows × 12 columns</p>
</div>
```

Great, the above code just left us with one clean and no duplicated rows of data. Now the records are down from `690` to `637`. Now lets check whether their still duplicates from the previous list of `patient_id` we had queried earlier, lets try the `1182404` `patient_id` string for that matter.

```python
# let's check whether the 1182404 patient_id still has duplication.
df.loc[df['patient_id'] == 1182404]
```

```html
<div>
	<style scoped>
		.dataframe tbody tr th:only-of-type {
			vertical-align: middle;
		}

		.dataframe tbody tr th {
			vertical-align: top;
		}

		.dataframe thead th {
			text-align: right;
		}
	</style>
	<table
		border="1"
		class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>patient_id</th>
				<th>clump_thickness</th>
				<th>cell_size_uniformity</th>
				<th>cell_shape_uniformity</th>
				<th>marginal_adhesion</th>
				<th>single_ep_cell_size</th>
				<th>bare_nuclei</th>
				<th>bland_chromatin</th>
				<th>normal_nucleoli</th>
				<th>mitoses</th>
				<th>class</th>
				<th>doctor_name</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>136</th>
				<td>1182404</td>
				<td>4.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>1</td>
				<td>2</td>
				<td>1</td>
				<td>2.0</td>
				<td>1.0</td>
				<td>1</td>
				<td>benign</td>
				<td>Dr. Lee</td>
			</tr>
		</tbody>
	</table>
</div>
```

```python
df.isnull().values.any()
```

False

# Visual Analysis

And they say, picture says a thousand words. And I couldn't agree more with the statement, we as a human easily absorb information, through graphs, colors and visualization, in contrast to just plain numbers. In this section, lets try to visualize our findings better.

There are numerous great visualization libraries out there for both Python and Pandas, but Ive been experimenting with Seaborn for awhile, and found them somewhat easier to implement to our objectives. Here are some of the benefit of having Seaborn as your library of choice for visualisation as taken from the official homepage:

> Seaborn aims to make visualization a central part of exploring and
> understanding data. Its dataset-oriented plotting \> functions operate
> on dataframes and arrays containing whole datasets and internally
> perform the necessary semantic mapping and statistical aggregation to
> produce informative plots.

Here is some of the functionality that seaborn offers:

- A dataset-oriented API for examining
  [relationships](https://seaborn.pydata.org/examples/scatter_bubbles.html#scatter-bubbles)
  between [multiple
  variables](https://seaborn.pydata.org/examples/faceted_lineplot.html#faceted-lineplot)
- Specialized support for using categorical variables to show
  [observations](https://seaborn.pydata.org/examples/jitter_stripplot.html#jitter-stripplot)
  or [aggregate
  statistics](https://seaborn.pydata.org/examples/pointplot_anova.html#pointplot-anova)
- Options for visualizing
  [univariate](https://seaborn.pydata.org/examples/distplot_options.html#distplot-options)
  or
  [bivariate](https://seaborn.pydata.org/examples/joint_kde.html#joint-kde)
  distributions and for
  [comparing](https://seaborn.pydata.org/examples/horizontal_boxplot.html#horizontal-boxplot)
  them between subsets of data
- Automatic estimation and plotting of [linear
  regression](https://seaborn.pydata.org/examples/anscombes_quartet.html#anscombes-quartet)
  models for different kinds
  [dependent](https://seaborn.pydata.org/examples/logistic_regression.html#logistic-regression)
  variables
- Convenient views onto the overall
  [structure](https://seaborn.pydata.org/examples/scatterplot_matrix.html#scatterplot-matrix)
  of complex datasets
- High-level abstractions for structuring [multi-plot
  grids](https://seaborn.pydata.org/examples/faceted_histogram.html#faceted-histogram)
  that let you easily build
  [complex](https://seaborn.pydata.org/examples/pair_grid_with_kde.html#pair-grid-with-kde)
  visualizations
- Concise control over matplotlib figure styling with several
  [built-in
  themes](https://seaborn.pydata.org/tutorial/aesthetics.html#aesthetics-tutorial)
- Tools for choosing [color
  palettes](https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial)
  that faithfully reveal patterns in your data

## Patients for Each Doctor?

Ever wonder how many patients each doctor handled from the dataset? We know for sure, we have 4 doctors from the dataset, but havent got some perfect ideas on how many patients each doctor is handling them. So why dont we try to visualize them, to see how many patients for each doctor needs to handle from the dataset?

```python
df['doctor_name'].value_counts()
```

doctor_name

- Dr. Doe 167
- Dr. Lee 165
- Dr. Smith 164
- Dr. Wong 141
  Name: count, dtype: int64

{.cell .code execution_count="25"}

```python
df['patient_id'].count()
```

637

```python
fig_dims = (12, 6)
fig, ax = plt.subplots(figsize = fig_dims)

ax.axes.set_title("How Many Patients Per Doctor", fontsize=14)
ax.set_xlabel('doctor_name',fontsize = 12)
ax.set_ylabel('Patients',fontsize = 12)

sns.set_style('whitegrid')
sns.countplot(x = 'doctor_name', palette = 'RdBu_r', data=df)
sns.despine()
```

- `Dr. Doe` :167
- `Dr. Lee` :165
- `Dr. Smith`:164
- `Dr. Wong` :141

## Class Cases For Each Doctor?

As mentioned on the earlier sections, we have a column name `class`, which basically contains the value of either `benign` and `malignant`. We wish to understand further whether a persons tumor is `malignant` (cancerous) or `benign` (not cancerous). With that being said, lets get down to business and try to visualize them further down below.

```python
class_by_doctor = df[("class")].value_counts()
class_by_doctor
```

```python
fig_dims = (12, 6)
fig, ax = plt.subplots(figsize=fig_dims)

ax.axes.set_title("Patient with Cancer Syndrome Per Doctor",fontsize=15)
ax.set_xlabel('X_axis',fontsize = 12)
ax.set_ylabel('Y_axis',fontsize = 12)

sns.despine()
sns.set_style('whitegrid')
sns.barplot(x="class", y="patient_id", hue="doctor_name", ci=None, palette='RdBu_r', data=df)
```

```python
# This time, let's do them horizontally

fig_dims = (12, 6)
fig, ax = plt.subplots(figsize=fig_dims)

ax.axes.set_title("Class of Patient Per Doctor",fontsize=15)
ax.set_xlabel('Doctor Name',fontsize = 12)
ax.set_ylabel('Patients',fontsize = 12)

sns.despine()
sns.set_style('whitegrid')
sns.barplot(x = "patient_id", y='class', hue="doctor_name", ci=None, palette='RdBu_r', data=df)
```

## Class Case Per Doctor?

As mentioned on the earlier sections, we have a column name `class`, which basically contains the value of `benign` and `malignant`. We wish to understand further whether a persons tumor is `malignant` (cancerous) or `benign` (not cancerous). With that being said, lets get down to business and try to visualize them further down below.

```python
fig_dims = (12, 6)
fig, ax = plt.subplots(figsize=fig_dims)

ax.axes.set_title("Class Per Doctor",fontsize=15)
ax.set_xlabel('Doctor Name',fontsize = 12)
ax.set_ylabel('Patients',fontsize = 12)

sns.despine()
sns.set_style('whitegrid')
sns.barplot(x='doctor_name', y='patient_id', hue="class", ci=None, palette='RdBu_r', data=df)
```

```python
fig_dims = (12, 6)
fig, ax = plt.subplots(figsize=fig_dims)

ax.axes.set_title("Horizontal Classes Per Doctor",fontsize=15)
ax.set_xlabel('Doctor Name',fontsize = 12)
ax.set_ylabel('Patients',fontsize = 12)

sns.despine()
sns.set_style('whitegrid')
sns.barplot(y='doctor_name', x='patient_id', hue="class", ci=None, palette='RdBu_r', data=df)
```

```python
df.isnull().values.any()
df.isnull().sum().sum()
```

# One Hot Encoding

Now that we've gone through the previous topic of visualizing our dataset, lets continue to the next section of preparing them in a way that our machine learning algorithms, by which will be using them near the end of this article, would be able to pick them up and run them through our `predictive model` easily. You may ask, "Of all the previous process, they're not enough?". Well apparently, it's not sufficient enough to meet the standards.

As among one of the challenges that were facing is still within the dataset itself. Well be better off by modifying them to meet the requirements. Our dataset still consist some categorical values in them, the `doctors_name` and `class` columns are two of good examples. And Machine Learning algorithm dont normally like them. We need to modify these two columns, so that it would make it easier and less confusing for the machine learning model to process through. I came across this [great
example](https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621) on how to deal with the similar situation.

## `doctor_name` column. {#doctor_name-column}

Lets first try to deal with the `doctor_name` column. This particular consist of 4 distinct values in them and how Pandas would handle them would probably as an object rather than an integer. Lets have our work around for this particular area. Will create another variable and call it `doctors_hotEncoded` and use the `get_dummies` method to transform them to an encoded one.

```python
doctors_hotEncoded = pd.get_dummies(df['doctor_name'])
```

```python
doctors_hotEncoded
```

```python
combined_doctors_hotEncoded_df = pd.concat([df, doctors_hotEncoded], axis=1)
```

```python
combined_doctors_hotEncoded_df
```

```python
# Now let's drop the 'doctor_name' varibale
combined_doctors_hotEncoded_df = combined_doctors_hotEncoded_df.drop(columns=['doctor_name'])
```

```python
# This is how it would look like.
combined_doctors_hotEncoded_df
```

```python
combined_doctors_hotEncoded_df.isnull().values.any()
combined_doctors_hotEncoded_df.isnull().sum().sum()
```

## `class` column. {#class-column}

```python
# How to convert benign & malingant to 0 and 1
change_class_numeric = {'benign':0, 'malignant':1}
combined_doctors_hotEncoded_df['class'] = combined_doctors_hotEncoded_df['class'].map(change_class_numeric)
```

```python
combined_doctors_hotEncoded_df
```

```python
# Making a new column based on a nuemrical calcualtion of other columns in the df
combined_doctors_hotEncoded_df['new_column'] = df.normal_nucleoli * df.mitoses
```

```python
combined_doctors_hotEncoded_df.head()
```

```python
combined_doctors_hotEncoded_df.isnull().values.any()
combined_doctors_hotEncoded_df.isnull().sum().sum()
```

# Feature Generation

This is among the crucial aspect area of `Machine Learning` model in the article, as this article [point out](https://towardsdatascience.com/data-wrangling-with-pandas-5b0be151df4e) an individual might be classified as having a cancer if meet the following condtion:

- Their `cell_size_uniformity` is greater than 5, and
- Their `cell_shape_uniformity` is greater than 5.

Based on this information, we could create another Feature from them.

```python
# Feature building:
def celltypelabel(x):
    if ((x['cell_size_uniformity'] > 5) & (x['cell_shape_uniformity'] > 5)):
        return('1')
    else:
        return('0')
```

Then we use the pandas apply function to run the `celltypelabel(x)`
function on the dataframe.

```python
combined_doctors_hotEncoded_df['cell_type_label'] = combined_doctors_hotEncoded_df.apply(lambda x: celltypelabel(x), axis=1)
```

```python
combined_doctors_hotEncoded_df[['patient_id', 'cell_type_label']]
```

```python
combined_doctors_hotEncoded_df
```

```python
combined_doctors_hotEncoded_df.isnull().values.any()
combined_doctors_hotEncoded_df.isnull().sum().sum()
```

```python
combined_doctors_hotEncoded_df.describe()
```

```python
combined_doctors_hotEncoded_df.isnull().values.any()
combined_doctors_hotEncoded_df.isnull().sum().sum()
```

```python
pd.crosstab(combined_doctors_hotEncoded_df['class'], combined_doctors_hotEncoded_df['cell_type_label'])
```

## Correlating Features

Heatmap of Correlation between different features:

> `Positive` = Positive correlation, i.e. increase in one feature will
> increase the other feature & vice-versa.`<br >`{=html} \> `Negative` =
> Negative correlation, i.e. increase in one feature will decrease the
> other feature & vice-versa.

In our case, we focus on which features have strong positive or negative correlation with the _Survived_ feature.

```python
plt.figure(figsize=(30,20))
plt.xlabel("Values on X axis")
plt.ylabel('Values on Y axis')
sns.heatmap(combined_doctors_hotEncoded_df.drop('patient_id',axis=1).corr(),
            xticklabels=True,
            vmax=0.6,
            square=True,
            annot=True)
```

# Updating Data Types

```python
combined_doctors_hotEncoded_df.info()
```

```python
# Let's try to change the datatypes of the following column in the dataset.
combined_doctors_hotEncoded_df['cell_type_label'] = combined_doctors_hotEncoded_df['cell_type_label'].astype('float64')
```

```python
# Let's try to change the datatypes of the following column in the dataset.
combined_doctors_hotEncoded_df['bare_nuclei'] = pd.to_numeric(combined_doctors_hotEncoded_df.bare_nuclei, errors='coerce')
```

```python
combined_doctors_hotEncoded_df.isnull().values.any()
```

```python
combined_doctors_hotEncoded_df.isnull().sum().sum()
```

```python
np.all(np.isfinite(combined_doctors_hotEncoded_df))
```

```python
combined_doctors_hotEncoded_df.isnull().sum().sum()
```

```python
# The bare_nuclei still has NaN or empty values in them?
combined_doctors_hotEncoded_df['bare_nuclei'].describe()
```

```python
# delete the empty rows
combined_doctors_hotEncoded_df.dropna(axis=0, how='any', inplace=True)
```

```python
np.any(np.isnan(combined_doctors_hotEncoded_df))
```

```python
combined_doctors_hotEncoded_df.isnull().sum().sum()
```

```python
np.all(np.isfinite(combined_doctors_hotEncoded_df))
```

```python
combined_doctors_hotEncoded_df.info()
```

```python
combined_doctors_hotEncoded_df.head()
```

# Spliting Dataset

```python
train, test = train_test_split(combined_doctors_hotEncoded_df, test_size=0.2)
train = pd.DataFrame(train)
test = pd.DataFrame(test)
```

```python
# Now that we've managed to split our main combined dataset into train and test dataset, let's test them.
train.head()
```

```python
test.head()
```

```python
train.info()
```

```python
test.info()
```

# Machine Learning

## Feature Selection

We drop unnecessary columns/features and keep only the useful ones for our experiment. Column _patient_id_ is only dropped from Train set because we need _patient_id_ in Test set while for running the experimentation.

```python
train = train.drop(['patient_id', 'new_column'], axis=1)
test = test.drop('cell_type_label', axis=1)
```

## Classification & Accuracy

Define training and testing set

```python
# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
```

```python
train.head()
```

```python
test.head()
```

```python
X_train = train.drop('cell_type_label', axis=1)
y_train = train['cell_type_label']
X_test = test.drop(["patient_id", "new_column"], axis=1).copy()

X_train.shape, y_train.shape, X_test.shape
```

```python
X_train.to_csv('train.csv', encoding='utf-8', index = False)
X_test.to_csv('test.csv', encoding='utf-8', index = False)
```

```python
X_train
```

```python
y_train
```

```python
X_test
```

## Logistic Regression

[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), or logit regression, or logit model is a regression model where the dependent variable (DV) is categorical. This article covers the case of a binary dependent variable---that is, where it can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analyzed in multinomial logistic regression, or, if the multiple categories are ordered, in ordinal logistic regression.

```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + '%')
```

## Support Vector Machine (SVM)

[Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) model is a Supervised Learning model used for classification and regression analysis. It is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. Suppose some given data points each belong to one of two classes, and the goal is to decide which class a new data point will be in. In the case of support vector machines, a data point is viewed as a $p$-dimensional vector (a list of $p$ numbers), and we want to know whether we can separate such points with a $(p-1)$-dimensional hyperplane.

When data are not labeled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. The clustering algorithm which provides an improvement to the support vector machines is called **support vector clustering** and is often used in industrial applications either when data are not labeled or when only some data are labeled as a preprocessing for a classification pass.

In the below code, [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) stands for Support Vector Classification.

```python
clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_svc)
```

## Linear SVM

Linear SVM is a SVM model with linear kernel.

In the below code, [LinearSVC](http://scikit learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) stands for Linear Support Vector Classification.

```python
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_linear_svc)
```

## $k$-Nearest Neighbors

[$k$-nearest neighbors algorithm (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is one of the simplest machine learning algorithms and is used for classification and regression. In both cases, the input consists of the $k$ closest training examples in the feature space. The output depends on whether $k$-NN is used for classification or regression:

- In _$k$-NN classification_, the output is a class membership. An
  object is classified by a majority vote of its neighbors, with the
  object being assigned to the class most common among its $k$ nearest
  neighbors ($k$ is a positive integer, typically small). If $k = 1$,
  then the object is simply assigned to the class of that single
  nearest neighbor.

- In _$k$-NN regression_, the output is the property value for the
  object. This value is the average of the values of its $k$ nearest
  neighbors.

```python
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn)
```

## Decision Tree

A [decision tree](https://en.wikipedia.org/wiki/Decision_tree) is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules.

```python
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)
```

### Random Forest

[Random forests](https://en.wikipedia.org/wiki/Random_forest) or **random decision forests** are an **ensemble learning method** for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for _decision trees habit of overfitting to their training set_.

[Ensemble methods](https://en.wikipedia.org/wiki/Ensemble_learning) use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.

```python
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)
```

## Gaussian Naive Bayes

[Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) are a family of simple probabilistic classifiers based on applying Bayes theorem with strong (naive) independence assumptions between the features.

[Bayes theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) (alternatively **Bayes law** or **Bayes rule**) describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if cancer is related to age, then, using Bayes theorem, a persons age can be used to more accurately assess the probability that they have cancer, compared to the assessment of the probability of cancer made without knowledge of the persons age.

Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.

```python
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb)
```

## Perceptron

[Perceptron](https://en.wikipedia.org/wiki/Perceptron) is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

```python
clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
print (acc_perceptron)
```

## Stochastic Gradient Descent (SGD)

[Stochastic gradient descent (https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (often shortened in **SGD**), also known as incremental gradient descent, is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions. In other words, SGD tries to find minima or maxima by iteration.

```python
clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print (acc_sgd)
```

## Confusion Matrix

A [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix), also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabelling one as another).

In predictive analytics, a table of confusion (sometimes also called a confusion matrix), is a table with two rows and two columns that reports the number of false positives, false negatives, true positives, and true negatives. This allows more detailed analysis than mere proportion of correct classifications (accuracy). Accuracy is not a reliable metric for the real performance of a classifier, because it will yield misleading results if the data set is unbalanced (that is, when the numbers of observations in different classes vary greatly). For example, if there were 95 cats and only 5 dogs in the data set, a particular classifier might classify all the observations as cats. The overall accuracy would be 95%, but in more detail the classifier would have a 100% recognition rate for the cat class but a 0% recognition rate for the dog class.

Heres another guide explaining [Confusion Matrix with example](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/).

$\begin{matrix} & Predicted Positive & Predicted Negative \\ Actual Positive & TP & FN \\ Actual Negative & FP & TN \end{matrix}$

In our (Titanic problem) case:

> **True Positive:** The classifier predicted _Survived_ **and** the
> passenger actually _Survived_.`<br />`{=html} \>**True Negative:** The
> classifier predicted _Not Survived_ **and** the passenger actually
> _Not Survived_.`<br />`{=html} \>**False Postiive:** The classifier
> predicted _Survived_ **but** the passenger actually _Not
> Survived_.`<br />`{=html} \>**False Negative:** The classifier
> predicted _Not Survived_ **but** the passenger actually _Survived_.

In the example code below, we plot a confusion matrix for the prediction of **_Random Forest Classifier_** on our training dataset. This shows how many entries are correctly and incorrectly predicted by our classifer.

```python
from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Cancerous', 'Not Cancerous']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Cancerous', 'True Not Cancerous']
predicted_class_names = ['Predicted Cancerous', 'Predicted Not Cancerous']

df_cnf_matrix = pd.DataFrame(cnf_matrix,
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent,
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)
```

## Comparing Models

Lets compare the accuracy score of all the classifier models used
above.

```python
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC',
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes',
              'Perceptron', 'Stochastic Gradient Decent'],

    'Score': [acc_log_reg, acc_svc, acc_linear_svc,
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb,
              acc_perceptron, acc_sgd]
    })

models.sort_values(by='Score', ascending=False)
```

From the above table, we can see that _Decision Tree_ and _Random
Forest_ classfiers have the highest accuracy score. Among these two, we
choose _Random Forest_ classifier as it has the ability to limit
overfitting as compared to _Decision Tree_ classifier.

# Create Prediction

{.cell .code editable="true" slideshow="{\"slide_type\":\"\"}" tags="[]"}

```python
submission = pd.DataFrame({
        "patient_id": test["patient_id"],
        "cell_type_label": y_pred_random_forest
    })

submission.to_csv('submission.csv', index=False)
```
