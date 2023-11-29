import json

# Загрузите данные из файлов old_data.json и data2.json
with open('data.json', 'r') as file1:
    data1 = json.load(file1)

with open('data2.json', 'r') as file2:
    data2 = json.load(file2)

# Объедините данные из двух файлов
combined_data = data1 + data2


# Отсортируйте данные по полю 'date' (unix)
sorted_data = sorted(combined_data, key=lambda x: x['date'])

# Удаление свойства 'liquidity' у каждого элемента
for item in sorted_data:
    if 'liquidity' in item:
        del item['liquidity']

# Сохранение обновленных данных в новый файл
with open('combined_data_no_liquidity.json', 'w') as outfile:
    json.dump(sorted_data, outfile, indent=4)
# Сохраните отсортированные данные в новый файл


print("Данные успешно объединены, отсортированы и сохранены в combined_data.json.")
