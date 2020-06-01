require(ggalluvial)

plotalluvial <- function(df, p){
  df <- df[order(df$subjectkey),]
  name_col <- paste('cluster_1', 'cluster_2')
  alluvdf <- data.frame('subjectkey'=df$subjectkey, 
                        'cluster_1'=df$cluster_subdomain, 'cluster_2'=df$cluster_domain)
  alluvdf$sex <- df$sex
  alluvdf <- alluvdf[order(alluvdf$sex),]
  is_alluvia_form(alluvdf)
  ggplot(alluvdf,
         aes(axis1 = cluster_1, axis2 = cluster_2)) +
    geom_alluvium(aes(fill=sex), width = 1/12) +
    geom_stratum(width = 1/12, fill = "black", color = "grey") +
    geom_label(stat = "stratum", infer.label = TRUE) +
    scale_x_discrete(limits = c("Subdomain", "Domain"), expand = c(.05, .05)) +
    scale_fill_brewer(type = "qual", palette = "Set1") +
    ggtitle(sprintf("Subjects relabeling at period %s", p))
}
data_folder_name <- '/Users/ilandi/PycharmProjects/ndar-stratification/out'

# P1
name_df <- 'imputed_data_P1.csv'
df <- read.table(file.path(data_folder_name, name_df),
                 sep=',',
                 header=TRUE,
                 as.is=TRUE)
plotalluvial(df, 'P1')

# P2
name_df <- 'imputed_data_P2.csv'
df <- read.table(file.path(data_folder_name, name_df),
                 sep=',',
                 header=TRUE,
                 as.is=TRUE)
plotalluvial(df, 'P2')
