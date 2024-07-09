# Genetic_Programming
The Genetic Programming Project utilizes an algorithm to determine the best clustering model and associated parameters for time series data. The algorithm uses data from the [National Grid ESO API](https://www.nationalgrideso.com/data-portal/api-guidance). This consists of several time series from Britain's and surrounding countries energy grid.

Step by Step Project Explanation: 

1. Required Imports 

2. API Calls
     
     Retrieving data with GET API requests. The resulting json data is normalized, datetime columns converted as required, unnecessary columns dropped, and dataframe sizes and nulls checked. These dataframes are concatenated and the time series are scaled using the StandardScaler - this is required for clustering since many of the clustering algorithms utilize distance and similarity for grouping. 

3. Dictionaries

     Dictionaries are used for the evolution functions, each dictionary acting as a holding place for the models and hyperparameters used throughout. 

3. Evolution Function

     Import the evolution function. This includes other functions including model instantiation, initial population generation, fitness scoring, population selection, crossover, and mutation. 

4. Running the Algorithm 

     The evolution function is run. This function returns the top 10 performing models, and also shows the output of each generation as it is running, showing the top model and associated score. 

## Usage

The bulk of the project is housed within an .ipynb file to provide insight into results. Directory: Genetic_Programming/genetic/Electricity_Genetic_Programming.ipynb.

A significant portion of development is found in the custom functions written for this project: Genetic_Programming/genetic/evolution_fns.py.

To get started, first clone the repository. This will download a copy of the repository in your current working directory. 

```python
$ git clone https://github.com/cgp9900/Genetic_Programming.git
```
