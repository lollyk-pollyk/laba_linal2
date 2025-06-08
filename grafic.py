import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 11)
accuracy = [40.4373, 70.7483, 82.7307, 86.3111, 89.0106,
            91.0309, 92.5518, 93.6927, 94.5678, 95.2322]
error = [59.5627, 29.2517, 17.2693, 13.6889, 10.9894,
         8.9691, 7.4482, 6.3073, 5.4322, 4.7678]

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, 'b-o', label='Точность')
plt.plot(epochs, error, 'r--s', label='Ошибка')
plt.title('Динамика обучения перцептрона', fontsize=16)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Процент (%)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(epochs)
plt.ylim(0, 100)
plt.savefig('combined_plot.png', dpi=300)
plt.show()