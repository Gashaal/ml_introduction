import pandas as pd

data = pd.read_csv('./titanic.csv', index_col='PassengerId')

sex = data['Sex'].value_counts()
print('Male: {}'.format(sex.get('male')))
print('Female: {}'.format(sex.get('female')))

survived = round(data['Survived'].value_counts(normalize=True).get(1) * 100, 2)
print('Survived: {}'.format(survived))

firstClass = round(data['Pclass'].value_counts(normalize=True).get(1) * 100, 2)
print('FirstClass: {}'.format(firstClass))

print('Mean age: {}'.format(round(data['Age'].mean(), 2)))
print('Median age: {}'.format(data['Age'].median()), 2)

print('Pearson: {0}'.format(round(data['SibSp'].corr(data['Parch']), 2)))

female = data.query('Sex == "female"')
names = []
for name in female['Name']:
    if "Mrs." in name:
        if '(' in name:
            names.append(name.split('Mrs.')[1].split('(')[1].split()[0])
        else:
            names.append(name.split('Mrs.')[1].strip())
    else:
        names.append(name.split(',')[1].split()[1])
print('Common female name: {}'.format(pd.Series(names).describe().top))