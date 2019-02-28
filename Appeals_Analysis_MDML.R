# Authors: Andrea Hassler, Jui Arvind Nerurkar, Emmanuel Bazov
# Load libraries
library(tidyverse)
library(tidytext)
library(tm)
library(ROCR)

######
# A) Import the data in the file recent_opinions.tsv as a tibble called ‘appeals.data’. Add an ‘opinion_id’ column to appeals.data, which gives each row a unique id. Load tidytext’s stop_words tibble (using the command data(stop_words)), and add the words in custom_words.txt to it to create a custom dictionary of stop words (you can set the lexicon as “custom” for these words). [5 pts]

# Read in tsv
appeals.data <- read_tsv('recent_opinions.tsv')

# Add unique ID
appeals.data <- appeals.data %>% mutate(opinion_id = c(1:nrow(appeals.data)))

# Load stop_words tibble
data(stop_words)

# Read in custom_words
custom_words <- read_delim('custom_words.txt', delim = '\n', col_names = FALSE)

# Add lexicon "custom"
custom_words_full <- tibble(word = custom_words$X1,
                            lexicon = "custom")

# Combine tibbles
custom_stop <- bind_rows(stop_words, custom_words_full)

######
# B) You will now build a simple bag-of-words classifier using the top 100 words (that are not stop words) in the corpus.


###
# a) Unnest the tokens in appeals.data and remove custom stop words. What are the 10 most common words in the entire corpus, and in each of the two circuits (not including stop words)? [5 pts]

# Unnest tokens
text_appeals.data <- appeals.data %>% unnest_tokens(word, text)

# Remove custom stop words
text_appeals.data <- text_appeals.data %>% anti_join(custom_stop)

# Top 10 most common words in entire corpus
text_appeals.data %>% count(word) %>% arrange(desc(n)) %>% slice(1:10)

# Top 10 most common words in 9th circuit court
text_appeals.data %>% filter(circuit == 'ninth') %>%
  count(word) %>% arrange(desc(n)) %>% slice(1:10)

# Top 10 most common words in the 5th circuit court
text_appeals.data %>% filter(circuit == 'fifth') %>%
  count(word) %>% arrange(desc(n)) %>% slice(1:10)


###
# b) Build a document-term tibble, where each row represents an opinion (there should be 16389 rows). There should be 102 columns: the circuit (code “fifth” as 1, and “ninth” as 0), the opinion_id, and the number of occurrences of the 100 most common words in the corpus (that are not stop words). For example, if a document contains the word “government” twice, the value in the “government” column for that document would be 2. Randomly shuffle the rows of this tibble, and split into a 50% training set and 50% test set. [10 pts]

# Create empty tibble
document_term <- tibble(opinion_id = appeals.data$opinion_id,
                        circuit = ifelse(appeals.data$circuit == "fifth", 1, 0))

# Top 100 most common words in entire corpus
top100 <- 
  text_appeals.data %>% 
  anti_join(custom_stop) %>% 
  count(word) %>% 
  arrange(desc(n)) %>% 
  slice(1:100) %>% 
  pull(word)

# Bind to tibble
document_term[top100] <- NA

# Loop to fill in tibble
for (i in 1:100) {
  document_term[3:102][i] <- 
    text_appeals.data %>% 
    group_by(opinion_id) %>% 
    summarize(sum(word == top100[i])) %>%
    pull(2)
}

# Create train and test sets
set.seed(12345)
train <- document_term %>% sample_frac(0.5)
test <- document_term %>% filter(!(opinion_id %in% train$opinion_id))


###
# c) Fit a logistic regression model on the training set that predicts the circuit as a function of all other predictors. If you got warning messages, what do they say? Compute the AUC of your model on the test set. Explain why your result is strange and which predictor is causing the strange result. [5 pts]

# Fit logistic regression model
model_partBc <- glm(circuit ~ . , data = train, family = binomial)

# Warning messages:
# 1: glm.fit: algorithm did not converge 
# 2: glm.fit: fitted probabilities numerically 0 or 1 occurred

# Compute AUC on the test set
test$predicted_probability <- predict(model_partBc, newdata = test, type = 'response')
test.pred <- prediction(test$predicted_probability, test$circuit)
test.perf <- performance(test.pred, "auc")
cat("The AUC score is", 100*test.perf@y.values[[1]], "\n")


###
# d) Drop the predictor referred to in part c) above, and refit your logistic regression model on the training set. What is your new AUC on the test set? What are the five smallest and five largest coefficients in this model? Give a precise interpretation of the largest model coefficient. [5 pts]

# Drop opinion_id and rerun regression
model_partBd <- update(model_partBc, . ~ . -opinion_id)
summary(model_partBd)

# Compute new AUC
test$predicted_probability <- predict(model_partBd, newdata = test, type = 'response')
test.pred <- prediction(test$predicted_probability, test$circuit)
test.perf <- performance(test.pred, "auc")
cat("The AUC score is", 100*test.perf@y.values[[1]], "\n")

# Five smallest coefficients
small_names <- sort(abs(coef(model_partBd)[-1]))[1:5] %>% names()
coef(model_partBd)[small_names]

# Five largest coefficients
large_names <- sort(abs(coef(model_partBd)[-1]), decreasing = T)[1:5] %>% names()
coef(model_partBd)[large_names]


######
# C) Repeat sub-parts a), b), and d) of part B) above, but this time, consider the top 100 bigrams instead of individual words. When you are removing stop words, make sure you remove bigrams that contain a stop word as either the first or second word in the bigram. [5 pts]

###
# a) Unnest the tokens in appeals.data and remove custom stop words. What are the 10 most common words in the entire corpus, and in each of the two circuits (not including stop words)? [5 pts]

# Unnest tokens for bigrams
bigrams_appeals.data <- appeals.data %>% unnest_tokens(bigram, text, token = 'ngrams', n=2)

# Remove custom stop words
bigrams_appeals.data <- bigrams_appeals.data %>%
  separate(bigram, c('word1', 'word2'), sep = " ") %>% 
  filter(!word1 %in% custom_stop$word) %>% 
  filter(!word2 %in% custom_stop$word) %>% 
  unite(bigram, word1, word2, sep = " ")

# Top 10 most common bigrams in entire corpus
bigrams_appeals.data %>% count(bigram) %>% arrange(desc(n)) %>% slice(1:10)

# Top 10 most common bigrams in 9th circuit court
bigrams_appeals.data %>% filter(circuit == 'ninth') %>%
  count(bigram) %>% arrange(desc(n)) %>% slice(1:10)

# Top 10 most common bigrams in the 5th circuit court
bigrams_appeals.data %>% filter(circuit == 'fifth') %>%
  count(bigram) %>% arrange(desc(n)) %>% slice(1:10)

###
# b) Build a document-term tibble, where each row represents an opinion (there should be 16389 rows). There should be 102 columns: the circuit (code “fifth” as 1, and “ninth” as 0), the opinion_id, and the number of occurrences of the 100 most common words in the corpus (that are not stop words). For example, if a document contains the word “government” twice, the value in the “government” column for that document would be 2. Randomly shuffle the rows of this tibble, and split into a 50% training set and 50% test set. [10 pts]

# Top 100 most common bigrams in entire corpus
bigrams_top100 <- 
  bigrams_appeals.data %>%
  separate(bigram, c('word1', 'word2'), sep = " ") %>% 
  filter(!word1 %in% custom_stop$word) %>% 
  filter(!word2 %in% custom_stop$word) %>% 
  unite(bigram, word1, word2, sep = " ") %>%  
  count(bigram) %>% 
  arrange(desc(n)) %>% 
  slice(1:100) %>% 
  pull(bigram)

# Filter to counts of top 100
bigrams_appeals <- 
  bigrams_appeals.data %>% 
  count(opinion_id, bigram) %>% 
  filter(bigram %in% bigrams_top100)

# Create document term matrix
dtm_bigrams <- bigrams_appeals %>% cast_dtm(document = opinion_id, term = bigram, value = n)
matrix <- as.matrix(dtm_bigrams)
temp <- as.tibble(matrix)

# Add back in opinion_id column
temp <- add_column(temp, opinion_id = dtm_bigrams[["dimnames"]][["Docs"]], .before = 1)

# Find opinion_id that do not contain top 100 and add back with NAs
missing_ids <- appeals.data$opinion_id[!(appeals.data$opinion_id %in% temp$opinion_id)]
temp <- add_row(temp, opinion_id = missing_ids)

# Make opinion_id numeric
temp <- temp %>% 
  mutate(
    opinion_id = as.numeric(opinion_id))

# Arrange by opinion_id
temp <- temp %>% arrange(opinion_id)

# Add back in circuits
temp <- temp %>% 
  add_column(circuit = ifelse(appeals.data$circuit == "fifth", 1, 0), .before = 2)

# Fill in zeros for NA values
temp <- temp %>% replace(., is.na(.), 0)

# Rename
bigrams_document_term <- temp

# Create train and test sets
set.seed(12345)
train_bigrams <- bigrams_document_term %>% sample_frac(0.5)
test_bigrams <- bigrams_document_term %>% filter(!(opinion_id %in% train_bigrams$opinion_id))

###
# d) Drop the predictor referred to in part c) above, and refit your logistic regression model on the training set. What is your new AUC on the test set? What are the five smallest and five largest coefficients in this model? Give a precise interpretation of the largest model coefficient. [5 pts]

# Drop opinion_id and rerun regression
model_partCd <- glm(circuit ~ . -opinion_id, data = train_bigrams, family = binomial)

# Compute new AUC
test_bigrams$predicted_probability <- predict(model_partCd, 
                                              newdata = test_bigrams, type = 'response')
test.pred_bigrams <- prediction(test_bigrams$predicted_probability, test_bigrams$circuit)
test.perf_bigrams <- performance(test.pred_bigrams, "auc")
cat("The AUC score is", 100*test.perf_bigrams@y.values[[1]], "\n")

# Five smallest coefficients
small_names_bigrams <- sort(abs(coef(model_partCd)[-1]))[1:5] %>% names()
coef(model_partCd)[small_names_bigrams]

# Five largest coefficients
large_names_bigrams <- sort(abs(coef(model_partCd)[-1]), decreasing = T)[1:5] %>% names()
coef(model_partCd)[large_names_bigrams]


######
# D) Repeat sub-parts b) and d) of part B) above, considering the top 100 bigrams and using the tf-idf value for each of the top 100 bigrams. Compute the tf-idf value for each bigram using the entire corpus of data, not just the training set. [5 pts]

###
# b) Build a document-term tibble, where each row represents an opinion (there should be 16389 rows). There should be 102 columns: the circuit (code “fifth” as 1, and “ninth” as 0), the opinion_id, and the number of occurrences of the 100 most common words in the corpus (that are not stop words). For example, if a document contains the word “government” twice, the value in the “government” column for that document would be 2. Randomly shuffle the rows of this tibble, and split into a 50% training set and 50% test set. [10 pts]

# Create document term matrix
dtm_bigrams <- bigrams_appeals %>% 
  cast_dtm(document = opinion_id, term = bigram, value = n, weighting = weightTfIdf)
matrix <- as.matrix(dtm_bigrams)
temp <- as.tibble(matrix)

# Add back in opinion_id column
temp <- add_column(temp, opinion_id = dtm_bigrams[["dimnames"]][["Docs"]], .before = 1)

# Find opinion_id that do not contain top 100 and add back with NAs
missing_ids <- appeals.data$opinion_id[!(appeals.data$opinion_id %in% temp$opinion_id)]
temp <- add_row(temp, opinion_id = missing_ids)

# Make opinion_id numeric
temp <- temp %>% 
  mutate(
    opinion_id = as.numeric(opinion_id))

# Arrange by opinion_id
temp <- temp %>% arrange(opinion_id)

# Add back in circuits
temp <- temp %>% 
  add_column(circuit = ifelse(appeals.data$circuit == "fifth", 1, 0), .before = 2)

# Fill in zeros for NA values
temp <- temp %>% replace(., is.na(.), 0)

# Rename
bigrams_document_term <- temp

# Create train and test sets
set.seed(12345)
train_bigrams <- bigrams_document_term %>% sample_frac(0.5)
test_bigrams <- bigrams_document_term %>% filter(!(opinion_id %in% train_bigrams$opinion_id))

###
# d) Drop the predictor referred to in part c) above, and refit your logistic regression model on the training set. What is your new AUC on the test set? What are the five smallest and five largest coefficients in this model? Give a precise interpretation of the largest model coefficient. [5 pts]

# Drop opinion_id and rerun regression
model_partDd <- glm(circuit ~ . -opinion_id, data = train_bigrams, family = binomial)

# Compute new AUC
test_bigrams$predicted_probability <- predict(model_partDd, 
                                              newdata = test_bigrams, type = 'response')
test.pred_bigrams <- prediction(test_bigrams$predicted_probability, test_bigrams$circuit)
test.perf_bigrams <- performance(test.pred_bigrams, "auc")
cat("The AUC score is", 100*test.perf_bigrams@y.values[[1]], "\n")

# Five smallest coefficients
small_names_bigrams <- sort(abs(coef(model_partDd)[-1]))[1:5] %>% names()
coef(model_partDd)[small_names_bigrams]

# Five largest coefficients
large_names_bigrams <- sort(abs(coef(model_partDd)[-1]), decreasing = T)[1:5] %>% names()
coef(model_partDd)[large_names_bigrams]


######
# E) Suppose you wanted to apply the model you fit in part D) to a single new opinion. Think through how you would do this (write a few sentences about your thoughts). Does part D) actually make sense as a strategy to build a classifier? If not, what is one way you could still use tf-idf values to build a classifier? [5 pts]

# See write-up

######
# F) Generate all trigrams, making sure you remove trigrams that contain a stop word as either of the three words in the trigram. Examine, for each circuit, the top 10 trigrams (by frequency in the corpus) that contain the word “supreme.” Write a few sentences about what you see (e.g., what are the different contexts in which the 5th vs. 9th circuit opinions are mentioning “supreme”?). [5 pts]

# Unnest tokens for bigrams
trigrams_appeals.data <- appeals.data %>% unnest_tokens(trigram, text, token = 'ngrams', n=3)

# Remove custom stop words
trigrams_appeals.data <- trigrams_appeals.data %>%
  separate(trigram, c('word1', 'word2', 'word3'), sep = " ") %>% 
  filter(!word1 %in% custom_stop$word) %>% 
  filter(!word2 %in% custom_stop$word) %>% 
  filter(!word3 %in% custom_stop$word) %>% 
  unite(trigram, word1, word2, word3, sep = " ")

#Examine top ten trigrams that contain the word supreme by circuit
trigrams_appeals.data %>% filter(circuit=='fifth', grepl('supreme', trigram) ) %>% count(trigram, sort=T) %>% slice(1:10)

trigrams_appeals.data %>% filter(circuit=='ninth', grepl('supreme', trigram) ) %>% count(trigram, sort=T) %>% slice(1:10)
