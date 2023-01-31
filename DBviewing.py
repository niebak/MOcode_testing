from code_functions import *
from tabulate import tabulate
DB = pd.read_excel('LUWfeature_example.xlsx')
stats = DB.describe()

print((stats.loc['std']))