

library(xlsx)
library(ggplot2)
library(mgcv)
library(caret) 
library(SiZer)
library(dplyr)
library(ggpmisc)
library(egg)

l_names <- c("nitrate_ammonification","nitrite_ammonification",
             "nitrification","denitrification",'nitrite_respiration',
             "nitrogen_fixation",'nitrate_respiration','nitrate_reduction')

l_renames <- c(expression(NO[3]^{'-'}~ammonification),expression(NO[2]^{'-'}~ammonification),
               'Ntrification','Denitrification',
               expression(NO[2]^{'-'}~respiration),'N fixation',
               expression(NO[3]^{'-'}~respiration),expression(NO[3]^{'-'}~reduction))

f_ornames <- c("Temperature","Precipitation","PM10","PM2.5",
               "Vehicles","Passengers","Goods","Tourists",
               "pH","DO","TDS","Chla","Na","Mg","Al","K","Ca",
               "TC","DOC","IC","TON",'NO3_N','NH4_N',
               "Shannon","Chao1","Niche_B")

f_renames <- c("Temperature (℃)","Precipitation (mm)",
               expression(paste(PM[10],' (μg/',m^3,')')),expression(paste(PM[2.5],' (μg/',m^3,')')),
               "Vehicles (10k)","Passengers (10k)","Goods (10kton)","Tourists (10k)",
               "pH","DO (mg/l)","TDS (mg/l)","Chla (μg/l)","Na (μg/l)","Mg (μg/l)","Al (μg/l)",
               "K (μg/l)","Ca (μg/l)","TC (mg/l)","DOC (mg/l)","IC (mg/l)","TON (mg/l)",
               expression(paste(NO[3]^'-','-N (mg/l)')),expression(paste(NH[4]^'+','-N (mg/l)')),
               "Shannon","Chao1","Niche B")

r2_p <- data.frame()
i=3 #nitrification
i=2 #nitrite_ammonification
i=6 #nitrogen_fixation
for (i in 1:length(l_names)) {

  shap_values <- read.xlsx('shap values.xlsx',sheetName = l_names[i])
  index=shap_values$NA.+1
  f_names <- colnames(shap_values)[-1]

  features <- read.xlsx('Features selected.xlsx',sheetName = l_names[i])[,-c(2,3)]
  features <- features[index,][,-1]

  shap_values <- shap_values[,-1]
  f_names <- colnames(shap_values)
  
  j=1
  for (j in c(1:length(f_names))){
    
    data <- data.frame('variable'=features[,j],'shap_values'=shap_values[,j])
    
    f_name <- f_renames[which(f_ornames==f_names[j])]
    
    temp_data <- scale(data$variable,center=T,scale=T)
    temp_index <- which((temp_data > 3)|(temp_data < -3))
    if (!is.na(temp_index[1])){
      data <- data[-temp_index,]
    }
    
    gam <- gam(shap_values~s(variable),data=data)
    p <- summary(gam)$s.pv
    p <- round(summary(gam)$s.pv,3)
    r2 <- round(summary(gam)$r.sq,3)
    temp <- data.frame('label'=l_names[i],'feature'=f_names[j],'r2'=r2,'p'=p)
    r2_p <- rbind(r2_p,temp)

    xscale <- max(data$variable)-min(data$variable)
    yscale <- max(data$shap_values)-min(data$shap_values)
    
    p <- ggplot(data=data,aes(x=variable,y=shap_values)) +
      geom_point(size=2,color='#0089FA') +
      geom_smooth(method='gam')+
      annotate(geom="text", x=0.9*max(data$variable), y=0.9*max(data$shap_values),
               label=as.character(as.expression(substitute(italic(R)^2~'='~b,list(b=r2)))),
               parse=TRUE,size=5)+
      annotate(geom="text", x=0.9*max(data$variable), y=0.65*max(data$shap_values),
               label=as.character(as.expression(substitute(italic(p)~'<'~0.05))),
               parse=TRUE,size=5)+
      scale_x_continuous(breaks=seq(25,115,15))+
      labs(x=f_name,y='Shap value',title=l_renames[i])+
      theme_bw()+
      theme(text=element_text(size=20,color='black'),
            axis.text = element_text(colour = 'black'),
            plot.title=element_text(size=15,hjust=0.5,color='black'))+
      # theme(plot.margin=unit(rep(1,4),'lines'))+
      theme(panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            panel.background=element_rect(fill = "transparent",colour = NA),
            plot.background=element_rect(fill = "transparent",colour = NA)) 
    p

    ggsave(paste0('plot/shap/gam/',l_names[i],'-',f_names[j],'.pdf'),
           egg::set_panel_size(p,width=unit(4, "in"), height=unit(2, "in")), 
           width=5,height=3.5,units='in',dpi=100)

  }
}

write.xlsx(r2_p,'r2_p_shap_gam.xlsx')

dev.off()
dev.new()


