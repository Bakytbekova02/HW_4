##  Сравните качество классификации (по правильности, точности, полноте или F-мере в зависимости от вашей задачи) посредством метода k ближайших соседей и логистической регрессии.

### Описание:
В данном коде мы реализуем две модели классификации - k ближайших соседей и логистическую регрессию. Мы также реализуем несколько метрик для оценки качества моделей, таких как точность, полнота и F1-мера.

### Набор данных
Создание набора данных

### Разделение данных
Разделение данных на обучающий и тестовый наборы вручную

### k ближайших соседей
Для модели k ближайших соседей мы реализуем класс `KNNClassifier`. Мы используем евклидово расстояние для определения ближайших соседей.

### Логистическая регрессия
Для модели логистической регрессии мы реализуем класс `LogisticRegression`. Мы используем градиентный спуск для обучения модели.

### Оценка качества
Мы используем метрики точности, полноты и F1-меры для оценки качества обеих моделей. Эти метрики реализованы вручную.
