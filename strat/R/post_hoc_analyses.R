require(ggplot2)
require(reshape2)
library(RColorBrewer)
require(ComplexHeatmap)
require(ggalluvial)
require(ggpubr)

#########################FUNCTIONS#############################
# Compare missing information
confounders <- function(df, p, lev){
  print(sprintf('Check missing data confounders period %s, %s', p, lev))

  if (lev=='subdomain'){
    df$cluster <- df$cluster_subdomain
    df$missing <- df$missing_subdomain} else{
      if (lev=='domain'){
      df$cluster <- df$cluster_domain
      df$missing <- df$missing_domain} else {
        df$cluster <- df$new_cluster
        df$missing <- df$missing_domain + df$missing_subdomain}
    }
  df$cluster <- as.factor(df$cluster)
  
  # Percentage of missing info per subject
  if (length(unique(df$cluster)) == 2){
    print(t.test(df$missing~df$cluster))
  } else{
    print(summary(aov(df$missing~df$cluster)))
    print(pairwise.t.test(df$missing, df$cluster))}
  
  #sex confounder
  sprintf('Compare sex for %s clusters at period %s', lev, p)
  tabsex_sub <- table(df$sex, df$cluster)
  print(tabsex_sub)
  print(chisq.test(tabsex_sub))
  
  #collection id
  sprintf('Compare collection_ids for %s clusters at period %s', lev, p)
  tabcollection_id_sub <- table(df$collection_id, df$cluster)
  print(tabcollection_id_sub)
  print(chisq.test(tabcollection_id_sub))
  
  #phenotype
  sprintf('Compare phenotypes for %s clusters at period %s', lev, p)
  tabpheno_sub <- table(df$phenotype, df$cluster)
  print(chisq.test(tabpheno_sub))
  
  #interview age
  sprintf("Compare age in months for %s clusters at time %s", lev, p)
  if (length(unique(df$cluster))==2){
    print(t.test(df$interview_age~df$cluster))
    mycomp <- list(as.character(sort(unique(df$cluster))))
  } else{
    print(summary(aov(interview_age~cluster, df)))
    print(pairwise.t.test(df$interview_age, df$cluster))
    mycomp <- list()
    i <- 1
    for (cl in 1:(length(sort(unique(df$cluster)))-1)){
      for (j in (cl+1):length(sort(unique(df$cluster)))){
        mycomp[[i]] <- c(as.character(as.character(sort(unique(df$cluster)))[cl]), 
                         as.character(as.character(sort(unique(df$cluster)))[j]))
        i <- i+1}}
  }
  print(ggplot(df, aes(x=as.factor(cluster), y=interview_age)) + 
          geom_boxplot() + 
          ggtitle(sprintf('Interview age in months vineland %s period %s', lev, p)) +
          xlab("Clusters") +
          ylab("Interview age") +
          geom_jitter(shape=16, position=position_jitter(0.2)) +
          stat_summary(fun=mean, geom="point", shape=21, size=4) + 
          stat_compare_means(comparisons = mycomp, method = "t.test", symnum.args = list(cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1), 
                                                                                         symbols = c("****", "***", "**", "*", "ns"))))
  
}


# Heatmaps for replicability
# Visualize subject distance between train/test and within clusters
replheat <- function(dist_mat_tr, dist_mat_ts, df_tr, df, p, lev){

  #TRAIN
  if (lev=='subdomain'){
    dist_mat_tr$cluster <- dist_mat_tr$cluster_subdomain} else {
      dist_mat_tr$cluster <- dist_mat_tr$cluster_domain
    }
  distdf_tr <- dist_mat_tr[order(apply(subset(dist_mat_tr, select=-grep('cluster', names(dist_mat_tr))), 1, mean)),
                         c(order(apply(subset(dist_mat_tr, select=-grep('cluster', names(dist_mat_tr))), 2, mean)), 
                           ncol(dist_mat_tr))]
  distdf_tr <- distdf_tr[order(distdf_tr$cluster), 
                         c(order(distdf_tr$cluster), 
                           ncol(distdf_tr))]
  clust_tr <- distdf_tr$cluster
  distmat_tr <- as.matrix(subset(distdf_tr, select=-grep('cluster', names(dist_mat_tr))))
  
  row.names(distmat_tr) <- row.names(distdf_tr)
  colnames(distmat_tr) <- names(distdf_tr)[1:(ncol(distdf_tr)-1)]
  
  colSide <- brewer.pal(9, "Set1")[3:9]
  col_v <- list(clusters = c())
  for (idx in sort(unique(clust_tr))){
    col_v$clusters <- c(col_v$clusters, colSide[idx])}
  names(col_v$clusters) <- as.character(sort(unique(clust_tr)))
  
  hTR <- Heatmap(distmat_tr,
                 heatmap_legend_param = list(
                   title = paste('VINELAND', '\ndist mat TR', sep=''), at = seq(min(distmat_tr),
                                                                                max(distmat_tr), 0.5)),
                 # name = paste(name_ins, '\ndist mat TR', sep=''),
                 show_row_names = FALSE,
                 show_column_names = FALSE,
                 show_row_dend = FALSE,
                 show_column_dend = FALSE,
                 cluster_rows = FALSE,
                 cluster_columns = FALSE,
                 # col = colorRampPalette(brewer.pal(8, "Blues"))(25),
                 left_annotation = HeatmapAnnotation(clusters=clust_tr,
                                                     col=col_v, which='row'),
                 top_annotation = HeatmapAnnotation(clusters=clust_tr,
                                                    col=col_v, which='column', 
                                                    show_legend = FALSE))
  # TEST
  if (lev=='subdomain'){
    dist_mat_ts$cluster <- dist_mat_ts$cluster_subdomain} else {
      dist_mat_ts$cluster <- dist_mat_ts$cluster_domain
    }
  distdf_ts <- dist_mat_ts[order(apply(subset(dist_mat_ts, select=-grep('cluster', names(dist_mat_ts))), 1, mean)),
                         c(order(apply(subset(dist_mat_ts, select=-grep('cluster', names(dist_mat_ts))), 2, mean)), 
                           ncol(dist_mat_ts))]
  distdf_ts <- distdf_ts[order(distdf_ts$cluster), 
                         c(order(distdf_ts$cluster), 
                           ncol(distdf_ts))]
  clust_ts <- distdf_ts$cluster
  distmat_ts <- as.matrix(subset(distdf_ts, select=-grep('cluster', names(dist_mat_ts))))
  row.names(distmat_ts) <- row.names(distdf_ts)
  colnames(distmat_ts) <- names(distdf_ts)[1:(ncol(distdf_ts)-1)]
  
  col_vts <- list(clusters = c())
  for (idx in sort(unique(clust_ts))){
    col_vts$clusters <- c(col_vts$clusters, colSide[idx])}
  names(col_vts$clusters) <- as.character(sort(unique(clust_ts)))
  
  hTS <- Heatmap(distmat_ts,
                 heatmap_legend_param = list(
                   title = paste('VINELAND', '\ndist mat TS', sep=''), at = seq(min(distmat_ts),
                                                                                max(distmat_ts), 0.5)),
                 # name = paste(name_ins, '\ndist mat TR', sep=''),
                 show_row_names = FALSE,
                 show_column_names = FALSE,
                 show_row_dend = FALSE,
                 show_column_dend = FALSE,
                 cluster_rows = FALSE,
                 cluster_columns = FALSE,
                 # col = colorRampPalette(brewer.pal(8, "Blues"))(25),
                 left_annotation = HeatmapAnnotation(clusters=clust_ts,
                                                     col=col_vts, which='row'),
                 top_annotation = HeatmapAnnotation(clusters=clust_ts,
                                                    col=col_vts, which='column', 
                                                    show_legend = FALSE))
  grid.newpage()
  title = sprintf('Feature Level %s RCV replication at period %s (train/test)', lev, p)
  grid.text(title, x=unit(0.5, 'npc'), y=unit(0.8, 'npc'), just='centre')
  pushViewport(viewport(x = 0, y = 0.75, width = 0.5, height = 0.5, just = c("left", "top")))
  grid.rect(gp = gpar(fill = "#00FF0020"))
  draw(hTR, newpage = FALSE)
  popViewport()
  
  pushViewport(viewport(x = 0.5, y = 0.75, width = 0.5, height = 0.5, just = c("left", "top")))
  grid.rect(gp = gpar(fill = "#0000FF20"))
  draw(hTS, newpage = FALSE)
  popViewport()
}


# Compare clusters
clust_comparison <- function(df, p, lev){
  print(sprintf('Comparing %s scores at period %s', lev, p))
  if (lev == 'subdomain'){
    df$cluster <- df$cluster_subdomain
    features <- names(df)[grep('vscore', names(df))]} else{
      if (lev == 'domain'){
      df$cluster <- df$cluster_domain
      features <- names(df)[grep('domain_', names(df))]} else {
        if (lev=='newcluster_subdomain'){
          df$cluster <- df$new_cluster
          features <- names(df)[grep('vscore', names(df))]} else {
            df$cluster <- df$new_cluster
            features <- names(df)[grep('domain_', names(df))]
        }
      }
    }
  df_long <- melt(subset(df, select=c('subjectkey', 'cluster', features)), 
                  id.vars=c('subjectkey', 'cluster'))
  df_long$cluster <- as.character(df_long$cluster)
  print(ggplot(df_long, aes(x=variable, y=value, fill=cluster)) +
          geom_boxplot() +
          facet_wrap(~variable, scale="free") +
          coord_cartesian(ylim = c(min(df_long$value), max(df_long$value))) +
          ggtitle(sprintf('%s features -- period %s', lev, p)))
  
  for (col in features){
    print(col)
    if (length(unique(df$cluster))==2){    
      print(t.test(df[, col]~df$cluster))
    } else{
      print(summary(aov(df[, col]~df$cluster)))
      print(pairwise.t.test(df[, col], df$cluster))}
  }
}
  

# Compare features within the same cluster
# Compare scores within clusters
feat_comparison <- function(df, p, lev){
  print(sprintf('Comparing clusters for %s features at period %s', lev, p))
  if (lev == 'subdomain'){
    df$cluster <- df$cluster_subdomain
    features <- names(df)[grep('vscore', names(df))]} else{
      if (lev == 'domain'){
        df$cluster <- df$cluster_domain
        features <- names(df)[grep('domain_', names(df))]} else {
          if (lev=='newcluster_subdomain'){
            df$cluster <- df$new_cluster
            features <- names(df)[grep('vscore', names(df))]} else {
              df$cluster <- df$new_cluster
              features <- names(df)[grep('domain_', names(df))]
            }
        }
    }
  df_long <- melt(subset(df, select=c('subjectkey', 'cluster', features)), 
                  id.vars=c('subjectkey', 'cluster'))
  df_long$cluster <- as.character(df_long$cluster)
  for (cl in sort(unique(df_long$cluster))){
    print(sprintf('Analyzing cluster %s', cl))
    print(pairwise.t.test(df_long$value[which(df_long$cluster==cl)], 
                          df_long$variable[which(df_long$cluster==cl)]))
  }
  print(ggplot(df_long, aes(x=cluster, y=value, fill=variable)) +
          geom_boxplot() +
          facet_wrap(~cluster, scale="free") +
          coord_cartesian(ylim = c(min(df_long$value), max(df_long$value))) +
          ggtitle(sprintf('%s features for each clusters -- period %s', lev, p)))}

# RUN comparisons
data_folder_name <- '/Users/ilandi/PycharmProjects/ndar-stratification/out'

# SUBDOMAIN - P1
name_df_tr <- 'imputed_data_P1_tr.csv'
name_df <- 'imputed_data_P1.csv'
name_dist_mat_tr <- 'vineland_distmatsubdomainTRP1.csv'
name_dist_mat_ts <- 'vineland_distmatsubdomainTSP1.csv'
df_tr <- read.table(file.path(data_folder_name, name_df_tr),
                 sep=',',
                 header=TRUE,
                 as.is=TRUE)
df <- read.table(file.path(data_folder_name, name_df),
                 sep=',',
                 header=TRUE,
                 as.is=TRUE)
distdf_tr <- read.table(file.path(data_folder_name, 
                                  name_dist_mat_tr),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)
distdf_ts <- read.table(file.path(data_folder_name, 
                                  name_dist_mat_ts),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)

confounders(df, 'P1', 'subdomain')
replheat(distdf_tr, distdf_ts, df_tr, df, 'P1', 'subdomain')
clust_comparison(df, 'P1', 'subdomain')
feat_comparison(df, 'P1', 'subdomain')

# DOMAIN - P1
name_dist_mat_tr <- 'vineland_distmatdomainTRP1.csv'
name_dist_mat_ts <- 'vineland_distmatdomainTSP1.csv'
distdf_tr <- read.table(file.path(data_folder_name, 
                                  name_dist_mat_tr),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)
distdf_ts <- read.table(file.path(data_folder_name, 
                                  name_dist_mat_ts),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)

confounders(df, 'P1', 'domain')
replheat(distdf_tr, distdf_ts, df_tr, df, 'P1', 'domain')
clust_comparison(df, 'P1', 'domain')
feat_comparison(df, 'P1', 'domain')

# NEWCLUSTER P1
confounders(df, 'P1', 'newcluster')
clust_comparison(df, 'P1', 'newcluster_subdomain')
feat_comparison(df, 'P1', 'newcluster_subdomain')

clust_comparison(df, 'P1', 'newcluster_domain')
feat_comparison(df, 'P1', 'newcluster_domain')

# SUBDOMAIN - P2
name_df_tr <- 'imputed_data_P2_tr.csv'
name_df <- 'imputed_data_P2.csv'
name_dist_mat_tr <- 'vineland_distmatsubdomainTRP2.csv'
name_dist_mat_ts <- 'vineland_distmatsubdomainTSP2.csv'
df_tr <- read.table(file.path(data_folder_name, name_df_tr),
                    sep=',',
                    header=TRUE,
                    as.is=TRUE)
df <- read.table(file.path(data_folder_name, name_df),
                 sep=',',
                 header=TRUE,
                 as.is=TRUE)
distdf_tr <- read.table(file.path(data_folder_name, 
                                  name_dist_mat_tr),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)
distdf_ts <- read.table(file.path(data_folder_name, 
                                  name_dist_mat_ts),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)

confounders(df, 'P2', 'subdomain')
replheat(distdf_tr, distdf_ts, df_tr, df, 'P2', 'subdomain')
clust_comparison(df, 'P2', 'subdomain')
feat_comparison(df, 'P2', 'subdomain')

# DOMAIN - P2
name_dist_mat_tr <- 'vineland_distmatdomainTRP2.csv'
name_dist_mat_ts <- 'vineland_distmatdomainTSP2.csv'
distdf_tr <- read.table(file.path(data_folder_name, 
                                  name_dist_mat_tr),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)
distdf_ts <- read.table(file.path(data_folder_name, 
                                  name_dist_mat_ts),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)

confounders(df, 'P2', 'domain')
replheat(distdf_tr, distdf_ts, df_tr, df, 'P2', 'domain')
clust_comparison(df, 'P2', 'domain')
feat_comparison(df, 'P2', 'domain')

# NEWCLUSTER P2
confounders(df, 'P2', 'newcluster')
clust_comparison(df, 'P2', 'newcluster_subdomain')
feat_comparison(df, 'P2', 'newcluster_subdomain')

clust_comparison(df, 'P2', 'newcluster_domain')
feat_comparison(df, 'P2', 'newcluster_domain')
