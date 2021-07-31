library(tidyverse)
library(ggplot2)

setwd("D:/PersonalProjects/NatHacks - Neuro Hackathon/RawEEG/left_blink")

readEEG <- function(file, raw = FALSE) {
  data <- read_csv(file) %>%
    rename("Time" = "Secondary Timestamp (Base)",
           "Offset" = "Secondary Timestamp (Remainder)")
  
  if (raw) {
    return(data)
  }
  
  data %>%
    mutate("timestamps" = NULL,
           Time = Time + Offset,
           Offset = NULL)
}

plotEEG <- function(data, electrode) {
  ggplot(data, aes(x = Time, y = eval(parse(text = electrode)))) +
    geom_line()
}

data <- readEEG("left_blink_var1_3.csv", raw = FALSE)
plotEEG(data, "AF8")

fucked <- c(3, 4, 5, 7, 8, 9)
