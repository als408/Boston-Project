# Ładowanie pakietów
library(MASS)
library(neuralnet)
library(rpart)
library(ggplot2)
library(corrplot)
library(gbm)
library(class)
library(randomForest)

# Usuwanie wartości NA
dane <- na.omit(Boston)
attach(dane)
set.seed(123)

# Podział danych na grupę uczącą i testującą
sample <- sample(1:nrow(dane), size = 0.7 * nrow(dane))
uczacy <- dane[sample, ]
testujacy <- dane[-sample, ]

# Standaryzacja zbioru uczącego
ucz <- scale(uczacy)

# Standaryzacja zbioru testowego
test <- scale(testujacy, center = attr(ucz, "scaled:center"), scale = attr(ucz, "scaled:scale"))

# Histogramy rozkładu zmiennych
hist(dane$crim, main="Rozkład CRIM", xlab="CRIM", col="blue", border="black")
hist(dane$rm, main="Rozkład RM", xlab="RM", col="blue", border="black")
hist(dane$dis, main="Rozkład DIS", xlab="DIS", col="blue", border="black")
hist(dane$lstat, main="Rozkład LSTAT", xlab="LSTAT", col="blue", border="black")

# Wykresy punktowe korelacji zmiennych z linią regresji
ggplot(dane, aes(x = crim, y = medv)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "blue", se = TRUE) +
  labs(title = "Wykres punktowy z linią regresji: CRIM vs MEDV",
       x = "Przestępczość (CRIM)",
       y = "Mediana wartości domów (MEDV)") +
  theme_minimal()

ggplot(dane, aes(x = rm, y = medv)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "blue", se = TRUE) +
  labs(title = "Wykres punktowy z linią regresji: RM vs MEDV",
       x = "Liczba pokoi (RM)",
       y = "Mediana wartości domów (MEDV)") +
  theme_minimal()

ggplot(dane, aes(x = lstat, y = medv)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "blue", se = TRUE) +
  labs(title = "Wykres punktowy z linią regresji: LSTAT vs MEDV",
       x = "Procent mieszkańców o niskim statusie społeczno-ekonomicznym (LSTAT)",
       y = "Mediana wartości domów (MEDV)") +
  theme_minimal()

ggplot(dane, aes(x = dis, y = medv)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "blue", se = TRUE) +
  labs(title = "Wykres punktowy z linią regresji: DIS vs MEDV",
       x = "Odległość od centrów zatrudnienia (DIS)",
       y = "Mediana wartości domów (MEDV)") +
  theme_minimal()

# Macierz korelacji w formie Heatmap
cor_matrix <- cor(ucz)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7, col = colorRampPalette(c("blue", "white", "red"))(200))

# Metoda 1: Regresja Liniowa
ucz <- data.frame(ucz)
test <- data.frame(test)
model_lin <- lm(medv ~ rm + lstat + crim + dis, data=ucz)
summary(model_lin)

pred_lin <- predict(model_lin, newdata=test)
lin_rmse <- sqrt(mean((pred_lin - test$medv)^2))
lin_rmse

plot(test$medv, pred_lin, main='Wykres modelu regresji liniowej', xlab='Prawdziwe wartości', ylab='Przewidywane wartości', col='blue')
abline(0, 1, col='orange')

# Metoda 2: Sieci Neuronowe
model_nn <- neuralnet(medv ~ rm + lstat + crim + dis, data=ucz, hidden=5, linear.output=TRUE, stepmax=1e6)
summary(model_nn)
plot(model_nn)

pred_nn <- predict(model_nn, newdata=test)
nn_rmse <- sqrt(mean((pred_nn - test$medv)^2))
nn_rmse

# Metoda 3: Drzewa Decyzyjne przy użyciu rpart
model_tree <- rpart(medv ~ rm + lstat + crim + dis, data=ucz, method="anova")
summary(model_tree)
printcp(model_tree)

# Przycinanie drzewa na podstawie wartości cp z najniższym błędem cross-validation (xerror)
optimal_cp <- model_tree$cptable[which.min(model_tree$cptable[,"xerror"]), "CP"]
prune_tree <- prune(model_tree, cp=optimal_cp)

# Wizualizacja przyciętego drzewa
plot(prune_tree, uniform=TRUE, main="Przycięte Drzewo Decyzyjne")
text(prune_tree, use.n=TRUE, all=TRUE, cex=.8)

pred_tree <- predict(prune_tree, newdata=test)
tree_rmse <- sqrt(mean((pred_tree - test$medv)^2))
tree_rmse

# Metoda 4: Random Forest (Random Forest działa na zasadzie tworzenia wielu drzew decyzyjnych i łączenia ich wyników.)
model_rf <- randomForest(medv ~ rm+dis+crim+lstat, data = ucz, ntree = 500, mtry = 4, importance = TRUE)

importance(model_rf) # Wyniki tego mówią o ważności danej zmiennej w modelu, %IncMSE pokazuje procentowy wzrost średniego błędu kwadratowego, 
varImpPlot(model_rf) # a IncNodePurity wzrost czystości węzłów (wyższa wartość oznacza, że cecha ta ma większy wpływ na tworzenie podziałów w drzewach decyzyjnych)

pred_rf <- predict(model_rf, newdata = test)
rf_rmse <- sqrt(mean((pred_rf - test$medv)^2))
rf_rmse

# Stworzenie ramki danych z rzeczywistymi i przewidywanymi wartościami
results_rf <- data.frame(
  Actual = test$medv,
  Predicted = pred_rf
)

# Wykres
ggplot(results_rf, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "orange", linetype = "dashed") +
  labs(
    title = "Predykcje modelu Random Forest vs Rzeczywiste wartości",
    x = "Rzeczywiste wartości medv",
    y = "Przewidywane wartości medv"
  ) +
  theme_minimal()

# Metoda 5: Gradient Boosting Machine (GBM) (ten model polega na łączeniu kilku słabych modeli np. drzew decyzyjnych w jeden silny model) # n.trees - model ma trenowac 1000 drzew decyzyjnych, interaction.depth - maksymalna glebokosc kazdego drzewa, shrinkage - tempo uczenia (mniejsze to dokladniejsze), cv.folds - okresla ilukrotna ma byc walidacja krzyzowa
model_gbm <- gbm(medv ~ rm+lstat+crim+dis, data = ucz, distribution = "gaussian", n.trees = 1000, interaction.depth = 4, shrinkage = 0.01, cv.folds = 5)
# Znalezienie najlepszej iteracji na podstawie cross-validation
best_iter <- gbm.perf(model_gbm, method = "cv")
pred_gbm <- predict(model_gbm, newdata = test, n.trees = best_iter)
gbm_rmse <- sqrt(mean((pred_gbm - test$medv)^2))
gbm_rmse

results_gbm <- data.frame(
  Actual = test$medv,
  Predicted = pred_gbm
)

# Stworzenie wykresu
ggplot(results_gbm, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "orange", linetype = "dashed") +
  labs(
    title = "Predykcje modelu GBM vs Rzeczywiste wartości",
    x = "Rzeczywiste wartości medv",
    y = "Przewidywane wartości medv"
  ) +
  theme_minimal()

# Porównanie wyników
results <- data.frame(
  Model = c("Regresja Liniowa", "Sieci neuronowe", "Drzewa Decyzyjne", "Random Forest", "GBM"),
  RMSE = c(lin_rmse, nn_rmse, tree_rmse, rf_rmse, gbm_rmse)
)
print(results)

# Wizualizacja porównania
ggplot(results, aes(x=Model, y=RMSE, fill=Model)) + 
  geom_bar(stat="identity") + 
  labs(title="Porównanie RMSE modeli", x="Model", y="RMSE") +
  theme_minimal()

# Wnioski końcowe
best_model <- results[which.min(results$RMSE), "Model"]
cat("Najlepszy model to:", best_model, "z najniższym RMSE:", min(results$RMSE), "\n")
# Wybieramy model z najniższą wartością RMSE, ponieważ jest uznawany za najlepszy, ze względu na to, że najlepiej przewiduje wartość medianowej wartości domów (medv) w zbiorze walidacyjnym.