import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'Cambria'
plt.rcParams['font.size'] = 14

df = sns.load_dataset('iris')
average_values = df.groupby('species').mean()
variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
num_variables = len(variables)
categories = average_values.index
num_categories = len(categories)

angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True), dpi=300)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

colors = ['lightcoral', 'lightblue', 'lightgreen', 'plum']
width = 2 * np.pi / num_categories

values_sepal_length = average_values['sepal_length'].values
values_sepal_length = np.concatenate((values_sepal_length, [values_sepal_length[0]]))
bars_sepal_length = ax.bar(angles, values_sepal_length, width=width, color=colors[0], edgecolor='black', linewidth=1.5,
                           label='sepal_length', align='center')
values_petal_length = average_values['petal_length'].values
values_petal_length = np.concatenate((values_petal_length, [values_petal_length[0]]))
bars_petal_length = ax.bar(angles, values_petal_length, width=width, color=colors[1], edgecolor='black', linewidth=1.5,
                           label='petal_length', align='center')
values_sepal_width = average_values['sepal_width'].values
values_sepal_width = np.concatenate((values_sepal_width, [values_sepal_width[0]]))
bars_sepal_width = ax.bar(angles, values_sepal_width, width=width, color=colors[2], edgecolor='black', linewidth=1.5,
                          label='sepal_width', align='center')
bars_petal_length_extra = ax.bar(angles[0], values_petal_length[0], width=width, color=colors[1], edgecolor='black',
                                 linewidth=1.5, label='petal_length', align='center')
values_petal_width = average_values['petal_width'].values
values_petal_width = np.concatenate((values_petal_width, [values_petal_width[0]]))
bars_petal_width = ax.bar(angles, values_petal_width, width=width, color=colors[3], edgecolor='black', linewidth=1.5,
                          label='petal_width', align='center')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

ax.legend(title='Variables', bbox_to_anchor=(1.1, 1.05), fontsize=12, title_fontsize=14, frameon=True, shadow=True)
plt.title('Overlay Coxcomb Chart of Iris Variables by Species', size=18, y=1.1)

# plt.show()

plt.savefig('plots/coxcomb_chart.png')
plt.close()
