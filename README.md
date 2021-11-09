# COVID-19 in Malaysia

## 📚Dataset

Data taken from the [Official Malaysia's COVID-19 data](https://github.com/MoH-Malaysia/covid19-public) as of [`11-09-2021`](https://github.com/MoH-Malaysia/covid19-public/commit/a9d2a11512d0943db02140a03486f6862df87107)

### Cases and Testing

1. [`cases_malaysia.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/cases_malaysia.csv): Daily recorded COVID-19 cases at country level, as of 1200 of date.
2. [`cases_state.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/cases_state.csv): Daily recorded COVID-19 cases at state level, as of 1200 of date.
3. [`clusters.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/clusters.csv): Exhaustive list of announced clusters with relevant epidemiological datapoints, as of 2359 of date of update.
4. [`tests_malaysia.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/tests_malaysia.csv): Daily tests (note: not necessarily unique individuals) by type at country level, as of 2359 of date.
5. [`tests_state.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/tests_malaysia.csv): Daily tests (note: not necessarily unique individuals) by type at state level, as of 2359 of date.

### Healthcare

1. [`pkrc.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/pkrc.csv): Flow of patients to/out of Covid-19 Quarantine and Treatment Centres (PKRC), with capacity and utilisation as of 2359 of date.
2. [`hospital.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/hospital.csv): Flow of patients to/out of hospitals, with capacity and utilisation as of 2359 of date.
3. [`icu.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/icu.csv): Capacity and utilisation of intensive care unit (ICU) beds as of 2359 of date.

### Deaths

1. [`deaths_malaysia.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/deaths_malaysia.csv): Daily deaths due to COVID-19 at country level, as of 1200 of date.
2. [`deaths_state.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/deaths_state.csv): Daily deaths due to COVID-19 at state level, as of 1200 of date.

### Static data

1. [`population.csv`](https://github.com/MoH-Malaysia/covid19-public/blob/main/static/population.csv): Total, adult (≥18), and elderly (≥60) population at state level.

## 💻Deployment

Our results are deployed on Heroku in the form of a Streamlit webapp.

Check out our project hosted on <a href="covid-19-msia-mining.herokuapp.com/" target="_blank">Heroku</a>! 

## 🔭Future Extensions

Checkout [COVID-19 Malaysia Cases and Vaccination](https://github.com/BingQuanChua/COVID-19-Msia-Cases-And-Vaccination)!