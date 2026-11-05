#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using I=std::int64_t;
static thread_local std::string error_text;
struct Tree {std::vector<int> f;std::vector<float> b;std::vector<double> leaves;};
struct Model {
 int nf=0,dim=0,log=0;double scale=1;std::vector<double> bias,corr;std::vector<Tree> trees;
 explicit Model(const char* path){
  std::ifstream s(path);std::string magic;int nt;s>>magic;
  if(magic!="CB_OBLIVIOUS_V1")throw std::runtime_error("Format modele invalide");
  s>>nf>>dim>>nt>>log;if(nf<=0||dim<=0||dim>1000||nt<0||nt>100000)throw std::runtime_error("Dimensions invalides");
  bias.resize(dim);corr.resize(dim);s>>scale;for(auto& v:bias)s>>v;for(auto& v:corr)s>>v;
  for(int j=0;j<nt;j++){Tree tr;int dep;s>>dep;if(dep<0||dep>16)throw std::runtime_error("Profondeur invalide");
   for(int k=0;k<dep;k++){int f;float b;s>>f>>b;if(f<0||f>=nf)throw std::runtime_error("Feature invalide");tr.f.push_back(f);tr.b.push_back(b);}
   tr.leaves.resize((1<<dep)*dim);for(auto& v:tr.leaves)s>>v;trees.push_back(std::move(tr));}
  if(!s)throw std::runtime_error("Modele tronque");
 }
 void predict(const float* x,double* y)const{
  std::fill(y,y+dim,0.);
  for(const auto& tr:trees){unsigned leaf=0;for(unsigned k=0;k<tr.f.size();k++)if(x[tr.f[k]]>tr.b[k])leaf|=1U<<k;const auto* v=tr.leaves.data()+leaf*dim;for(int k=0;k<dim;k++)y[k]+=v[k];}
  for(int k=0;k<dim;k++){y[k]=y[k]*scale+bias[k];if(log)y[k]=std::expm1(y[k]);y[k]=std::max(0.,y[k]*corr[k]);}
 }
};
struct Lot {I ready,expires;int qty;};struct Need {I at;int qty;};
struct Meat {
 std::deque<Lot> hot,cooking;std::deque<Need> needs;
 I demand=0,cooked=0,expired=0,served=0,missing=0,wait=0,launches=0,last=-100000000000LL;
 int hotq=0,pipeq=0,backlog=0;
};
struct Engine {
 std::array<Meat,3> ms;int lead,ttl,interval;std::array<int,3> caps,batches;std::array<double,3> safety3;double safety,disp;bool automatic=true;I now=std::numeric_limits<I>::min();
 Engine(int l,int life,const int* c,const int* b,double z,int iv,double d):lead(l),ttl(life),interval(iv),safety(z),disp(d){
  if(l<=0||life<=0||iv<0||z<0||d<0)throw std::runtime_error("Parametres invalides");
  for(int k=0;k<3;k++){caps[k]=c[k];batches[k]=b[k];safety3[k]=z;if(c[k]<=0||b[k]<=0||b[k]>c[k])throw std::runtime_error("Capacite invalide");}}
 void fulfil(Meat& m,I t){
  while(!m.needs.empty()&&!m.hot.empty()){auto& n=m.needs.front();auto& l=m.hot.front();int q=std::min(n.qty,l.qty);n.qty-=q;l.qty-=q;m.hotq-=q;m.backlog-=q;m.served+=q;m.wait+=q*(t-n.at);if(!n.qty)m.needs.pop_front();if(!l.qty)m.hot.pop_front();}}
 void advance(I t){
  if(t<now)throw std::runtime_error("Horloge reculee");
  while(true){I next=std::numeric_limits<I>::max();for(auto& m:ms){if(automatic&&!m.cooking.empty())next=std::min(next,m.cooking.front().ready);if(!m.hot.empty())next=std::min(next,m.hot.front().expires);}if(next>t)break;
   for(auto& m:ms){while(!m.hot.empty()&&m.hot.front().expires<=next){m.expired+=m.hot.front().qty;m.hotq-=m.hot.front().qty;m.hot.pop_front();}
    while(automatic&&!m.cooking.empty()&&m.cooking.front().ready<=next){auto l=m.cooking.front();m.cooking.pop_front();m.pipeq-=l.qty;m.cooked+=l.qty;m.hotq+=l.qty;m.hot.push_back(l);}fulfil(m,next);}}
  now=t;
 }
 void order(I t,const int* q){for(int k=0;k<3;k++)if(q[k]<0)throw std::runtime_error("Quantite negative");advance(t);for(int k=0;k<3;k++){auto& m=ms[k];m.demand+=q[k];m.missing+=std::max(0,q[k]-m.hotq);if(q[k]){m.needs.push_back({t,q[k]});m.backlog+=q[k];fulfil(m,t);}}}
 void completed(I t,int k,int q,I cooked){
  if(k<0||k>=3||q<=0||cooked>t)throw std::runtime_error("Lot invalide");advance(t);auto& m=ms[k];
  if(!automatic){int remove=std::min(q,m.pipeq);m.pipeq-=remove;while(remove&&!m.cooking.empty()){auto& l=m.cooking.front();int n=std::min(remove,l.qty);l.qty-=n;remove-=n;if(!l.qty)m.cooking.pop_front();}}
  m.cooked+=q;if(cooked+ttl<=t){m.expired+=q;return;}
  Lot l{cooked,cooked+ttl,q};auto it=std::upper_bound(m.hot.begin(),m.hot.end(),l.expires,[](I e,const Lot& x){return e<x.expires;});m.hot.insert(it,l);m.hotq+=q;fulfil(m,t);
 }
 void tick(I t,const double* f3,const double* f5,int allow,int* launched){
  advance(t);std::fill(launched,launched+3,0);
  for(int k=0;k<3;k++){auto& m=ms[k];double mu=allow?std::max(0.,f5[k]):0.,early=allow?std::max(0.,f3[k]):0.;int exp=0;for(const auto& l:m.hot)if(l.expires<=t+lead)exp+=l.qty;
   double supply=m.hotq-exp+std::min(double(exp),early)+m.pipeq;double target=mu+safety3[k]*std::sqrt(disp*mu)+m.backlog;int need=int(std::ceil(std::max(0.,target-supply)-1e-9));
   if((!allow&&!m.backlog)||need<=0||t-m.last<interval)continue;
   int q=((need+batches[k]-1)/batches[k])*batches[k];q=std::min(q,((caps[k]-m.pipeq)/batches[k])*batches[k]);
   if(q>0){m.cooking.push_back({t+lead,t+lead+ttl,q});m.pipeq+=q;m.launches++;m.last=t;launched[k]=q;}}
 }
 void state(double* out)const{for(int k=0;k<3;k++){const auto& m=ms[k];double* v=out+12*k;v[0]=m.hotq;v[1]=m.pipeq;v[2]=m.backlog;v[3]=m.demand;v[4]=m.cooked;v[5]=m.expired;v[6]=m.served;v[7]=m.missing;v[8]=m.wait;v[9]=m.launches;v[10]=m.hot.empty()?0:m.hot.front().expires;v[11]=m.cooking.empty()?0:m.cooking.front().ready;}}
};
static double cumulative(const double* p,int k,double h){
 const double hs[6]={0,1,3,5,15,30};double ys[6]={0,p[k],p[3+k],p[6+k],p[9+k],p[12+k]};
 for(int i=1;i<6;i++)ys[i]=std::max(ys[i-1],ys[i]);
 for(int i=1;i<6;i++)if(h<=hs[i])return ys[i-1]+(ys[i]-ys[i-1])*(h-hs[i-1])/(hs[i]-hs[i-1]);
 return ys[5]+(ys[5]-ys[4])*(h-30)/15;
}
extern "C" {
const char* core_error(){return error_text.c_str();}
void* model_create(const char* p){try{return new Model(p);}catch(const std::exception& e){error_text=e.what();return nullptr;}}
void model_free(void* p){delete static_cast<Model*>(p);}
int model_batch(void* p,const float* x,int n,double* y){try{auto* m=static_cast<Model*>(p);for(int i=0;i<n;i++)m->predict(x+i*m->nf,y+i*m->dim);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
void* engine_create(int l,int ttl,const int* caps,const int* batch,double z,int iv,double d){try{return new Engine(l,ttl,caps,batch,z,iv,d);}catch(const std::exception& e){error_text=e.what();return nullptr;}}
void engine_free(void* p){delete static_cast<Engine*>(p);}
void engine_auto(void* p,int enable){static_cast<Engine*>(p)->automatic=enable;}
void engine_safety(void* p,const double* z){for(int k=0;k<3;k++)static_cast<Engine*>(p)->safety3[k]=z[k];}
int engine_order(void* p,I t,const int* q){try{static_cast<Engine*>(p)->order(t,q);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
int engine_tick(void* p,I t,const double* f3,const double* f5,int allow,int* launched){try{static_cast<Engine*>(p)->tick(t,f3,f5,allow,launched);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
int engine_completed(void* p,I t,int k,int q,I cooked){try{static_cast<Engine*>(p)->completed(t,k,q,cooked);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
int engine_state(void* p,double* out){try{static_cast<Engine*>(p)->state(out);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
int simulate(I start,I end,const I* times,const int* quantities,int norders,const double* predictions,int nminutes,int lead,int ttl,const int* caps,const int* batches,const double* safety,int interval,double dispersion,int reactive,double* totals){
 try{
  Engine e(lead,ttl,caps,batches,0.,interval,dispersion);for(int k=0;k<3;k++)e.safety3[k]=safety[k];int cursor=0;std::array<double,3> recent{},expected{},partial{};std::deque<std::array<double,3>> hist,predhist;I previous=-1;
  for(I t=start;t<end+lead+ttl+1;t++){
   I mi=(t-start)/60;int second=(t-start)%60;
   if(mi!=previous){
    if(previous>=0){hist.push_back(partial);std::array<double,3> pr{};for(int k=0;k<3;k++)pr[k]=previous<nminutes?predictions[previous*15+k]:0;predhist.push_back(pr);if(hist.size()>3){hist.pop_front();predhist.pop_front();}}
    partial.fill(0);recent.fill(0);expected.fill(0);for(const auto& a:hist)for(int k=0;k<3;k++)recent[k]+=a[k];for(const auto& a:predhist)for(int k=0;k<3;k++)expected[k]+=a[k];previous=mi;
   }
   double f3[3]={0},f5[3]={0};bool allow=false;
   auto refresh=[&](){if(mi<nminutes&&t<end){const double* p=predictions+mi*15;allow=p[6]+p[7]+p[8]>.05;
    for(int k=0;k<3;k++){double fraction=second/60.,ratio=1.;if(reactive){double prior=std::max(2.,p[k]*2.);ratio=(recent[k]+partial[k]+prior)/(expected[k]+p[k]*fraction+prior);ratio=std::clamp(ratio,.35,3.5);}f3[k]=std::max(0.,cumulative(p,k,3+fraction)-cumulative(p,k,fraction))*ratio;f5[k]=std::max(0.,cumulative(p,k,5+fraction)-cumulative(p,k,fraction))*ratio;}}
   };
   while(cursor<norders&&times[cursor]<=t){const int* q=quantities+3*cursor;e.order(t,q);for(int k=0;k<3;k++)partial[k]+=q[k];cursor++;refresh();int launched[3];e.tick(t,f3,f5,allow,launched);}
   refresh();
   int launched[3];e.tick(t,f3,f5,allow,launched);
  }
  e.state(totals);if(cursor!=norders)throw std::runtime_error("Commandes hors periode");return 0;
 }catch(const std::exception& ex){error_text=ex.what();return -1;}
}
}
