import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

st.title("Анализ датасета 'Crimes in US Communities'")

df = pd.read_csv('crimedata.csv')

st.write(df)
st.markdown('''
<p align="center">
  Датасет 'Crimes in US Communities' 
</p>
''', unsafe_allow_html=True)

st.markdown(''' ## Корреляционная матрица''')


corr_method = st.selectbox('Метод подсчета корреляции?',
                      (
                           'spearman: Формула Спирмена',
                           'pearson: Формула Пирсона (стандартная)'
                      ))

crim_type = st.selectbox('Будете использовать показатель насильственных или ненасильственных преступлений?',
                        (
                            'ViolentCrimesPerPop: показатель насильственных преступлений',
                            'nonViolPerPop: показатель ненасильственных преступлений'
                        ))

chart_na = st.selectbox('Что делать с NA?',
                     (
                         'Удалить',
                         'Заполнить средним по штатам'
                     ))

color_2 = st.selectbox('Какой цвет barplot хотите использовать?',
                     (
                         'black',
                         'blue',
                         'gray',
                         'green',
                         'yellow'
                     ))

border = st.selectbox('Убрать показатели с корреляции меньше 0.1?',
                        (
                            'Yes',
                            'No'
                        ))

@st.cache
def fig_2(corr_method, chart_na, border, color_2, crim_type, db):

    state = db['state']
    db = db.select_dtypes(exclude=[object])
    db['state'] = state

    if chart_na == 'Удалить':
        db = db.dropna()
    else:
        db = db.groupby("state").transform(lambda x: x.fillna(x.mean()))

    corr_method = corr_method.split(': ')[0]
    crim_type = crim_type.split(': ')[0]
    db.sort_values(crim_type, ignore_index=True, inplace=True)



    if crim_type == 'nonViolPerPop':
        dataset = db.corr(method=corr_method).iloc[-1]
        x_axis = dataset.values
        y_axis = dataset.index
    elif crim_type == 'ViolentCrimesPerPop':
        dataset = db.corr(method=corr_method).iloc[-2]
        x_axis = dataset.values
        y_axis = dataset.index

    if border == 'Yes':
        dataset = dataset[abs(dataset) > 0.01]
        x_axis = dataset.values
        y_axis = dataset.index


    return [x_axis, y_axis, color_2, crim_type]


if st.checkbox('Показать график'):
    figure_2 = fig_2(corr_method, chart_na, border, color_2, crim_type, df)
    fig = plt.figure(figsize=(20, 150))
    plt.title('График №2')
    f_2 = sns.barplot(x=figure_2[0],
                 y=figure_2[1],
                 color=figure_2[2])
    f_2.bar_label(f_2.containers[0])
    f_2.set(xlabel=figure_2[3], ylabel='Variables')


    st.pyplot(fig)

st.markdown(''' Чем выше корреляция (зависимость целевой переменной от конкретной объясняющей переменной), тем выше 
зависимость. Это наталкивает на мысль посмотреть на переменные с наибольшей корреляцией. 
 
 Одной из переменных с наибольшей корреляцией является процент американских африканцев в совокупном населении. Другие 
 расы из датасета также имеют существенную корреляцию, поэтому исследуем соответствующие переменные. 
 Рассматривать только расу не имеет смысла, ибо часто рассовый аспект типичного нарушителя имеет под собой 
 другие факторы. Например, статус мигранта, который при массовом проявлении характеризуется низким доход, по сравнению с
 коренными жителями. Людям не остается ничего, кроме как начать совершать преступления. 
 
 Таким образом, рассмотрим доход на душу населения (по расе) и попытаемся найти штат с наименьшим доходом для
 дальнейшего исследования.     
 ''')

st.markdown(''' ## Доход на душу населения в разрезе расы''')

race_heritage = st.selectbox('Информацию о какой группе будете исследовать?',
                      (
                          'asian',
                          'african american',
                          'caucasian',
                          'hispanic'
                      ))

info_type = st.selectbox('Что хотите вывести?',
                        (
                            'PerCap: доход на душу населения',
                            'PctPolic: процент полицийских из данной группы',
                            'racepct: процент национальности в совокупном населении'
                        ))


color_3 = st.selectbox('Какой цвет barplot  хотите использовать?',
                     (
                         'black',
                         'blue',
                         'gray',
                         'green',
                         'yellow'
                     ))


chart_na_2 = st.selectbox('Что делать с NaN?',
                     (
                         'Удалить',
                         'Заполнить средним по штатам'
                     ))

def fig_3(race_heritage, info_type, color_3, chart_na_2, db):

    var = info_type.split(': ')[0]
    if var == 'PerCap':
        if race_heritage == 'asian':
            column = 'AsianPerCap'
        elif race_heritage == 'african american':
            column = 'blackPerCap'
        elif race_heritage == 'caucasian':
            column = 'whitePerCap'
        elif race_heritage == 'hispanic':
            column = 'HispPerCap'
    elif var == 'PctPolic':
        if race_heritage == 'asian':
            column = 'PctPolicAsian'
        elif race_heritage == 'african american':
            column = 'PctPolicBlack'
        elif race_heritage == 'caucasian':
            column = 'PctPolicWhite'
        elif race_heritage == 'hispanic':
            column = 'PctPolicHisp'
    elif var == 'racepct':
        if race_heritage == 'asian':
            column = 'racePctAsian'
        elif race_heritage == 'african american':
            column = 'racepctblack'
        elif race_heritage == 'caucasian':
            column = 'racePctWhite'
        elif race_heritage == 'hispanic':
            column = 'racePctHisp'
    state = db['state']
    db = db.select_dtypes(exclude=[object])
    db['state'] = state

    if chart_na_2 == 'Удалить':
        db = db.dropna()
    else:
        db[column] = db[[column, 'state']].groupby("state").transform(lambda x: x.fillna(x.mean()))

    data = db[[column, 'state']].groupby('state').agg('mean')[column]
    y_axis = data.values
    x_axis = data.index
    mean_value = db['perCapInc'].mean()

    return [x_axis, y_axis, color_3, mean_value, chart_na_2, var]


figure_3 = fig_3(race_heritage, info_type, color_3, chart_na_2, df)

fig = plt.figure(figsize=(30, 15))
plt.title('График №3')
f_3 = sns.barplot(x=figure_3[0],
             y=figure_3[1],
             color=figure_3[2])
f_3.bar_label(f_3.containers[0])
f_3.set(xlabel='state', ylabel='mean')

ax2 = f_3.twinx()
if figure_3[5] == 'PerCap' and figure_3[4] == 'Удалить':
    ax2.plot(np.random.uniform(figure_3[3],figure_3[3],size=14), color='r')
elif figure_3[5] == 'PerCap':
    ax2.plot(np.random.uniform(figure_3[3], figure_3[3], size=49), color='r')
ax2.grid(False)
st.pyplot(fig)

if st.checkbox('Показать график насильственных преступлений по штатам'):
    fig = plt.figure(figsize=(30, 130))
    plt.title('График №4')
    f_4 = sns.barplot(x=df[['state', 'ViolentCrimesPerPop']].groupby('state').agg('mean')['ViolentCrimesPerPop'].values,
                 y=df[['state', 'ViolentCrimesPerPop']].groupby('state').agg('mean')['ViolentCrimesPerPop'].index)
    f_4.bar_label(f_4.containers[0])
    f_4.set(xlabel='state', ylabel='mean crim')
    st.pyplot(fig)


st.markdown(''' Судя по полученным данным корреляция возникает не только при преобладании какой-то конкретной 
расы, но и при недостаточном уровне дохода. Это видно, например, для штатов LA, AL, AK для african
american или штат WV и доход asian (отрицательная корреляция исходного показателя в отличие от african american). Кроме
того, отметим, что процент полицеских конкретной расы имеет близкие результаты с  общим показателем 
расы в составе населения. 
 
 Необходимо подробнее рассмотреть переменные бедности (положительная корреляция с целевой переменной), а 
 также переменные благополучия (отрицательная корреляция с целевой переменной) 
 ''')

st.markdown(''' ## Переменные благополучия и переменные бедности''')

var_poor = st.selectbox('Переменная бедности',
                        (
                            'pctWPubAsst: percentage of households with public assistance income in 1989',
                            'PctPopUnderPov: percentage of people under the poverty level',
                            'PctLess9thGrade: percentage of people 25 and over with less than a 9th grade education',
                            'PctUnemployed: percentage of people 16 and over, in the labor force, and unemployed',
                            'PctKidsBornNeverMar: percentage of kids born to never married '
                        ))

low_border = st.slider('Нижняя граница количества преступлений', 0, 8000, 10)

target_var = st.selectbox('Насильственные или ненасильственные преступления?',
                          (
                              'ViolentCrimesPerPop: показатель насильственных преступлений',
                              'nonViolPerPop: показатель ненасильственных преступлений'
                          ))

show_count_crime = st.selectbox('Показать количество преступлений',
                                (
                                    'Да',
                                    'Нет'
                                ))

def fig_5(var_poor, low_border, target_var, show_count_crime, db):

    target_var = target_var.split(': ')[0]

    var_poor = var_poor.split(': ')[0]
    db[var_poor] = db[[var_poor, 'state']].groupby("state").transform(lambda x: x.fillna(x.mean()))
    state = db['state']
    db = db.select_dtypes(exclude=[object])
    db['state'] = state

    dataset_0 = db[[var_poor, 'state']].groupby('state').agg('mean')
    y_axis_0 = dataset_0[var_poor].values
    x_axis_0 = dataset_0[var_poor].index

    if show_count_crime == 'Да':
        db[target_var] = db[[target_var, 'state']].groupby('state').transform(lambda x: x.fillna(x.mean()))
        dataset_1 = db[[target_var, 'state']].groupby('state').agg('mean')
        dataset_1 = dataset_1[dataset_1 > low_border]
        x_axis_1 = dataset_1.index
        y_axis_1 = dataset_1.values
    else:
        x_axis_1 = None
        y_axis_1 = None

    return [x_axis_0, y_axis_0, x_axis_1, y_axis_1, low_border, show_count_crime]

figure_5 = fig_5(var_poor, low_border, target_var, show_count_crime, df)

fig = plt.figure(figsize=(30, 15))
plt.title('График №5')
f_5 = sns.barplot(x=figure_5[0],
             y=figure_5[1],
             color='black')
f_5.bar_label(f_5.containers[0])
f_5.set(xlabel='state', ylabel='mean')


if figure_5[-1] == 'Да':
    ax2 = f_5.twinx()
    ax2.plot(figure_5[3], color='r')
    ax2.grid(False)

st.pyplot(fig)


st.markdown(''' Переменные бедности действительно показывают схожую динамику с количеством преступлений. Далее необходимо 
проверить переменные благополучия.''')

var_well = st.selectbox('Переменная благополучия',
                        (
                            'pctWInvInc: percentage of households with investment / rent income in 1989',
                            'medFamInc: median family income',
                            'PctKids2Par: percentage of kids in family housing with two parents',
                            'PctEmploy: percentage of people 16 and over who are employed',
                            'PctHousOwnOcc: percent of households owner occupied'
                        ))

high_border = st.slider('Верхняя граница количества преступлений', 0, 4000, 10)

target_var_1 = st.selectbox('Насильственные или ненасильственныее преступления?',
                          (
                              'ViolentCrimesPerPop: показатель насильственных преступлений',
                              'nonViolPerPop: показатель ненасильственных преступлений'
                          ))

show_count_crime_1 = st.selectbox('Показать количество преступлений?',
                                (
                                    'Да',
                                    'Нет'
                                ))

def fig_6(var_well, high_border, target_var_1, show_count_crime_1, db):

    target_var_1 = target_var_1.split(': ')[0]

    var_well = var_well.split(': ')[0]
    db[var_well] = db[[var_well, 'state']].groupby("state").transform(lambda x: x.fillna(x.mean()))
    state = db['state']
    db = db.select_dtypes(exclude=[object])
    db['state'] = state

    dataset_0 = db[[var_well, 'state']].groupby('state').agg('mean')
    y_axis_0 = dataset_0[var_well].values
    x_axis_0 = dataset_0[var_well].index

    if show_count_crime_1 == 'Да':
        db[target_var_1] = db[[target_var_1, 'state']].groupby('state').transform(lambda x: x.fillna(x.mean()))
        dataset_1 = db[[target_var_1, 'state']].groupby('state').agg('mean')[target_var_1]
        dataset_1 = dataset_1[dataset_1 < high_border]
        x_axis_1 = dataset_1.index
        y_axis_1 = dataset_1.values
    else:
        y_axis_0 = dataset_0[var_well].values
        x_axis_0 = dataset_0[var_well].index
        x_axis_1 = None
        y_axis_1 = None

    return [x_axis_0, y_axis_0, x_axis_1, y_axis_1, high_border, show_count_crime_1]

figure_6 = fig_6(var_well, high_border, target_var_1, show_count_crime_1, df)

fig = plt.figure(figsize=(30, 15))
plt.title('График №6')
f_6 = sns.barplot(x=figure_6[0],
             y=figure_6[1],
             color='black')
f_6.bar_label(f_6.containers[0])
f_6.set(xlabel='state', ylabel='mean')


if figure_6[-1] == 'Да':
    ax2 = f_6.twinx()
    ax2.plot(figure_6[3], color='r')
    ax2.grid(False)

st.pyplot(fig)

st.markdown(''' Можно заметить, что есть слабая обратная зависимость. При высоком уровне преступности есть обратная 
зависимость от переменных благополучия. Таким образом, графики позволяют сделать вывод, что основной объясняющей 
переменной является бедность, представленная положительными и отрицательными переменными. Для более наглядного понимания
рассмотрим распределения данных переменных и преступлений.''')

vars = st.selectbox('Какой фактор?',
                    (
                        'pctWInvInc: percentage of households with investment / rent income in 1989',
                        'PctKids2Par: percentage of kids in family housing with two parents',
                        'PctHousOwnOcc: percent of households owner occupied',
                        'PctPopUnderPov: percentage of people under the poverty level',
                        'PctKidsBornNeverMar: percentage of kids born to never married '
                    ))

target_var_2 = st.selectbox('Насильственные или ненасильственные преступления?',
                          (
                              'ViolentCrimesPerPop: показатель насильственныхx преступлений',
                              'nonViolPerPop: показатель ненасильственных преступлений'
                          ))

density = st.checkbox('Показать границу плотности (kernel density function)')

def fig_7(vars, target_var_2, db):

    target_var_2 = target_var_2.split(': ')[0]
    var = vars.split(': ')[0]

    dataset_0 = db[var]
    dataset_1 = db[target_var_2]

    return [dataset_0, dataset_1, var, target_var_2]

figure_7 = fig_7(vars, target_var_2,  df)

st.set_option('deprecation.showPyplotGlobalUse', False)

fig, axs = plt.subplots(2, 1, figsize=(14, 14))
if density:
    sns.histplot(data=figure_7[0], kde=True, color="skyblue", ax=axs[0])
    sns.histplot(data=figure_7[1], kde=True, color="olive", ax=axs[1])
else:
    sns.histplot(data=figure_7[0], kde=False, color="skyblue", ax=axs[0])
    sns.histplot(data=figure_7[1], kde=False, color="olive", ax=axs[1])

st.pyplot()

st.markdown(''' Как и говорилось в предыдущих частях, основным фактором является бедность. Показатель с наибольшей 
 корреляцией (процент детей, рожденных вне брака), а также процент людей, относящихся к бедным, имеют такое же 
 распределение как и целевая переменная. Поэтому необходимо работать именно с этим фактором для дальнейшего исследования.
 В свою очередь показатели благосостояния имеют нормальное распределение (по графику), которое достаточно хорошо отображает
распределение целевой переменной.''')