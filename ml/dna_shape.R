args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: dna_shape.R input.fa output_dir expected_sequence_count")
}

fasta_path <- args[[1L]]
output_dir <- args[[2L]]
expected_count <- as.integer(args[[3L]])

suppressPackageStartupMessages(library(DNAshapeR))
feature_names <- c("MGW", "Roll", "ProT", "HelT")
predictions <- DNAshapeR::getShape(fasta_path, shapeType = feature_names)

as_sequence_rows <- function(values, n_sequences, feature_name) {
  if (is.list(values) && !is.data.frame(values) && !is.matrix(values)) {
    if (length(values) == n_sequences &&
        all(vapply(values, is.atomic, logical(1)))) {
      values <- do.call(rbind, lapply(values, as.numeric))
    } else {
      values <- as.matrix(values)
    }
  } else {
    values <- as.matrix(values)
  }

  if (nrow(values) == n_sequences) {
    return(values)
  }
  if (ncol(values) == n_sequences) {
    return(t(values))
  }
  stop(sprintf(
    "%s output has dimensions %s; cannot align %d input sequences",
    feature_name,
    paste(dim(values), collapse = " x "),
    n_sequences
  ))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
for (feature_name in feature_names) {
  if (is.null(predictions[[feature_name]])) {
    stop(sprintf("DNAshapeR did not return the %s feature", feature_name))
  }

  feature_matrix <- as_sequence_rows(
    predictions[[feature_name]],
    expected_count,
    feature_name
  )
  storage.mode(feature_matrix) <- "double"
  write.table(
    feature_matrix,
    file = file.path(output_dir, paste0(feature_name, ".tsv")),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE,
    na = "NaN"
  )
}
