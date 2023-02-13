
### log-density of the multivariate normal distribution


rm(list=ls())

ldmvnorm<-function(X,mu,Sigma,iSigma=solve(Sigma),dSigma=det(Sigma)) 
{
  Y<-t( t(X)-mu)
  sum(diag(-.5*t(Y)%*%Y%*%iSigma))  -
  .5*(  prod(dim(X))*log(2*pi) +     dim(X)[1]*log(dSigma) )
                                 
}
###


### sample from the multivariate normal distribution
rmvnorm<-function(n,mu,Sigma)
{
  p<-length(mu)
  res<-matrix(0,nrow=n,ncol=p)
  if( n>0 & p>0 )
  {
    E<-matrix(rnorm(n*p),n,p)
    res<-t(  t(E%*%chol(Sigma)) +c(mu))
  }
  res
}
###


### sample from the Wishart distribution
rwish<-function(n,nu0,S0)
{
  sS0 <- chol(S0)
  S<-array( dim=c( dim(S0),n ) )
  for(i in 1:n)
  {
     Z <- matrix(rnorm(nu0 * dim(S0)[1]), nu0, dim(S0)[1]) %*% sS0
     S[,,i]<- t(Z)%*%Z
  }
  S[,,1:n]
}
###

# DATA: 
library("readxl")


Y=read_excel("Vaccination data _ Groups.xlsx") 

### average values per group of the infection rate  
m = length(t(unique(Y[,1])))
n<-sv<-ybar<-rep(NA,m) 
for(j in 1:m) 
{ 
  ybar[j]<- mean(t(Y[Y[,1]==j,4]))   #mean(Y[[j]]) # mean infection
  sv[j]<-var((Y[Y[,1]==j,4]))           #std infection
  n[j]<-sum(Y[,1]==j)           #data point per group
}



group<-Y$ID #?
unique(group) #

X<-list() ; 
for(j in 1:m) 
{
  xj<-Y$`Total people fully vaccinated /population` [Y[,1]==j] #Y[,1] ? la prima colonna, contiene l'indice della scuola 
  #xj<-(xj-mean(xj))
  X[[j]]<-cbind( rep(1,n[j]), xj  )
}
       


x11()
plot(Y$`Total people fully vaccinated /population`,Y$`New cases`,xlab='People fully vax',ylab='New Cases')
dev.off()


S2.LS<-BETA.LS<-NULL
for(j in 1:m) {
  fit<-lm((t(t(Y[Y[,1]==j,4]))) ~ t(t((X[[j]][,2]))) ) 
  BETA.LS<-rbind(BETA.LS,c(fit$coef)) 
  S2.LS<-c(S2.LS, summary(fit)$sigma^2) 
                } 
x11()
par(mar=c(2.75,2.75,.5,.5),mgp=c(1.7,.7,0))
par(mfrow=c(1,3))
plot( range(Y$`Total people fully vaccinated /population`),range(((Y[,4]))),type="n",xlab="Vaccination ", 
   ylab=" infection rate")
for(j in 1:m) {    abline(BETA.LS[j,1],BETA.LS[j,2],col="gray")  }

BETA.MLS<-apply(BETA.LS,2,mean)
abline(BETA.MLS[1],BETA.MLS[2],lwd=2)

# MIDDLE panel
plot(n,BETA.LS[,1],xlab="sample size",ylab="intercept") 
 # intercepts of the different regression lines vs group sizes
abline(h= BETA.MLS[1],col="black",lwd=2)

# RIGHT panel
plot(n,BETA.LS[,2],xlab="sample size",ylab="slope") 
#slopes of the different regression lines vs group sizes
abline(h= BETA.MLS[2],col="black",lwd=2)

sum(BETA.LS[,2]>0)

dev.off()

####################################################################
## HIERARCHICAL REGRESSION MODEL - LMM LINEAR MIXED effects MODEL ##
####################################################################
p<-dim(X[[1]])[2]
# mu0, the prior expectation of theta is fixed equal to 
# the average of the corresponding (frequentist) 
# regression parameters  - 
# the matrix Lambda0 is the empirical covariance matrix of 
#        these 100 estimates
#
theta<-mu0<-apply(BETA.LS,2,mean)
nu0<-1 ; s2<-s20<-mean(S2.LS)
eta0<-p+2 ; #cos? la prior per la matrice Sigma ? diffusa
L0=matrix(nrow=2,ncol=2)
L0[1,1]=cov(BETA.LS)[1,1]
L0[1,2]=cov(BETA.LS)[1,2]
L0[2,1]=cov(BETA.LS)[2,1]
L0[2,2]=cov(BETA.LS)[2,2]

####  Inizializzazione della MC (Gibbs Sampler)
Sigma<-S0<-L0 #<-as.matrix(cov(BETA.LS)); 
BETA<-BETA.LS
THETA.b<-S2.b<-NULL
iL0<-solve(L0) ; iSigma<-solve(Sigma)
Sigma.ps<-matrix(0,p,p)
SIGMA.PS<-NULL
BETA.ps<-BETA*0
BETA.pp<-NULL
set.seed(1)
mu0[2]+c(-1.96,1.96)*sqrt(L0[2,2]) #prior IC per theta_2 (la slope media)



##### Gibbs Sampler cycle
for(s in 1:1000) {  #1000
  ##update beta_j 
  for(j in 1:m) 
  {  
    Vj<-solve( iSigma + t(X[[j]])%*%X[[j]]/s2 )
    Ej<-Vj%*%( iSigma%*%theta + t(X[[j]])%*%t(t(Y[Y[,1]==j,4]))/s2 )
    BETA[j,]<-rmvnorm(1,Ej,Vj) 
  } 
  ##

  ##update theta
  Lm<-  solve( iL0 +  m*iSigma )
  mum<- Lm%*%( iL0%*%mu0 + iSigma%*%apply(BETA,2,sum))
  theta<-t(rmvnorm(1,mum,Lm))
  ##

  ##update Sigma
  mtheta<-matrix(theta,m,p,byrow=TRUE)
  iSigma<-rwish(1, eta0+m, solve( S0+t(BETA-mtheta)%*%(BETA-mtheta) )  ) 
  ##

  ##update s2
  RSS<-0
  for(j in 1:m) { RSS<-RSS+sum( (Y[Y[,1]==j,4]-X[[j]]%*%BETA[j,] )^2 ) }
  s2<-1/rgamma(1,(nu0+sum(n))/2, (nu0*s20+RSS)/2 )
  ##
  ##store results
  if(s%%10==0) 
  { 
    cat(s,s2,"\n")
    S2.b<-c(S2.b,s2);THETA.b<-rbind(THETA.b,t(theta))
    Sigma.ps<-Sigma.ps+solve(iSigma) ; BETA.ps<-BETA.ps+BETA
    SIGMA.PS<-rbind(SIGMA.PS,c(solve(iSigma)))
    BETA.pp<-rbind(BETA.pp,rmvnorm(1,theta,solve(iSigma)) )
  }
  ##
}




##### Convergence diagnostics
library(coda)
effectiveSize(S2.b)        # sigma^2
effectiveSize(THETA.b[,1]) #theta_1
effectiveSize(THETA.b[,2])  #theta_2

apply(SIGMA.PS,2,effectiveSize)  #\Sigma

x11()
par(mfrow=c(2,2))
tmp<-NULL;for(j in 1:dim(SIGMA.PS)[2]) { tmp<-c(tmp,acf(SIGMA.PS[,j])$acf[2]) }

acf(S2.b)
acf(THETA.b[,1])
acf(THETA.b[,2])

dev.off()



x11()
par(mar=c(3,3,1,1),mgp=c(1.75,.75,0))
par(mfrow=c(1,2))

plot(density(THETA.b[,2],adj=2),xlim=range(BETA.pp[,2]), 
      main="",xlab="slope parameter",ylab="posterior density",lwd=2)
lines(density(BETA.pp[,2],adj=2),col="gray",lwd=2)
legend( -3 ,1.0 ,legend=c( expression(theta[2]),expression(tilde(beta)[2])), 
        lwd=c(2,2),col=c("black","gray"),bty="n") 

quantile(THETA.b[,2],prob=c(.025,.5,.975))

mean(BETA.pp[,2]<0) 

BETA.PM<-BETA.ps/1000
plot( range(Y$`Total people fully vaccinated /population`),range(((Y[,4]))),type="n",xlab="Vaccination", 
      ylab="infection rate")
for(j in 1:m) {    abline(BETA.PM[j,1],BETA.PM[j,2],col="gray")  }
abline( mean(THETA.b[,1]),mean(THETA.b[,2]),lwd=2 )


#dev.off()

windows()
par(mar=c(3,3,1,1),mgp=c(1.75,.75,0))
par(mfrow=c(1,2))
plot( range(Y$`Total people fully vaccinated /population`),range(((Y[,4]))),type="n",xlab="Total people vaccinated / population", 
   ylab="infection rate",main="Frequentist LS within-group regression lines")
for(j in 1:m) {    abline(BETA.LS[j,1],BETA.LS[j,2],col="gray")  }

BETA.MLS<-apply(BETA.LS,2,mean)
abline(BETA.MLS[1],BETA.MLS[2],lwd=2)

x11()
plot(range(Y$`Total people fully vaccinated /population`),range(((Y[,4]))),type="n",xlab="Total people vaccinated / population",
   ylab="infection rate",main="Bayesian estimates - hierarchical model")
for(j in 1:m) {    abline(BETA.PM[j,1],BETA.PM[j,2],col="gray")  }
abline( mean(THETA.b[,1]),mean(THETA.b[,2]),lwd=2 )


#####################################################################################
