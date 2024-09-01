import warnings

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Cambria'

df = sns.load_dataset('iris')
df_melted = pd.melt(df[['sepal_length', 'sepal_width']])

sepal_length_color = "#5b95c2"
sepal_width_color = "#ff9a42"

plt.figure(figsize=(10, 6))
sns.violinplot(x='variable', y='value', data=df_melted, inner="box",
               palette={"sepal_length": sepal_length_color, "sepal_width": sepal_width_color})
sns.boxplot(x='variable', y='value', data=df_melted, width=0.1, showcaps=True, boxprops={'facecolor': 'none'},
            whiskerprops={'color': 'black'}, capprops={'color': 'black'}, medianprops={'color': 'red'},
            showfliers=False)

handle_length = lines.Line2D([], [], color=sepal_length_color, marker='s', markersize=10, label='Sepal Length')
handle_width = lines.Line2D([], [], color=sepal_width_color, marker='s', markersize=10, label='Sepal Width')
plt.legend(handles=[handle_length, handle_width], title='Feature', loc='best')

plt.title('Violin Plot: Distribution of Sepal Length and Width', fontsize=14)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Length', 'Width'])
plt.grid(True, linestyle='--', alpha=0.6)

# plt.show()

plt.savefig('plots/violin_plot.png', dpi=300)
plt.close()
