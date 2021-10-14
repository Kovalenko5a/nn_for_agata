#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <vector>
#include <iomanip>



namespace AGATAGeFEM

{

#ifndef _EventStartToken_
#define _EventStartToken_
  const unsigned long int EventStartToken = 0xFFFFFFFF;
#endif


#ifndef _OneTimeStep_
#define _OneTimeStep_
  /**
     One "sampling point" for all segments at the same time.
  */
  template <typename T> 
  struct OneTimeStepT{
    double t;
    T Q[37];
  };

  typedef OneTimeStepT<short> OneTimeStep;
#endif


#ifndef _APulseShape_
#define _APulseShape_
  /**
     A class to keep a complete set of pulse shapes for a point
  */
  template <typename T>
  class APulseShapeT {
  public:
    APulseShapeT() : operatormap(18446744073709551615ULL) {;}
    APulseShapeT(const APulseShapeT& tcp) : 
      operatormap(18446744073709551615ULL) 
    {*this=tcp;}
    ~APulseShapeT() {;}
    std::ostream& write(std::ostream& os);
    std::istream& read(std::istream& os);
    template <typename T2> APulseShapeT<T>& 
    operator=(const APulseShapeT<T2>& rhs);
    APulseShapeT& operator+=(const APulseShapeT& rhs);
    APulseShapeT& operator-=(const APulseShapeT& rhs);
    const APulseShapeT operator+(const APulseShapeT& rhs);
    const APulseShapeT operator-(const APulseShapeT& rhs);
    const APulseShapeT operator*(const double& rhs);
    const APulseShapeT operator/(const double& rhs);
    const APulseShapeT operator*(const float& rhs);
    const APulseShapeT operator/(const float& rhs);
    const APulseShapeT operator*(const int& rhs);
    const APulseShapeT operator/(const int& rhs);
    const APulseShapeT operator*(const short& rhs);
    const APulseShapeT operator/(const short& rhs);
    inline void clear() {Qs.clear();}
    inline void resize(unsigned int i) {Qs.resize(i);}
    inline OneTimeStepT<T> & operator[](int i) {return Qs[i];}
    inline const OneTimeStepT<T> & operator[](int i) const {return Qs[i];}
    std::vector<double> operator()(double t,int *ll=0);
    //    std::vector<short> & operator()(const double &t);
    inline unsigned int size() {return Qs.size();}
    inline typename std::vector<OneTimeStepT<T> >::iterator begin() 
    {return Qs.begin();}
    inline typename std::vector<OneTimeStepT<T> >::iterator end() 
    {return Qs.end();}
    inline typename std::vector<OneTimeStepT<T> >::iterator back() 
    {return Qs.back();}
    inline float getx(){return x;}
    inline float gety(){return y;}
    inline float getz(){return z;}
    inline int getnumberoftimesteps() {return numberoftimesteps;}
    inline const unsigned int size() const {return Qs.size();} 
    inline const float getx() const {return x;}
    inline const float gety() const {return y;}
    inline const float getz() const {return z;}
    inline const int getnumberoftimesteps() const 
    {return numberoftimesteps;}
    inline void setx(float val){x=val;}
    inline void sety(float val){y=val;}
    inline void setz(float val){z=val;}
    inline void setnumberoftimesteps(unsigned int val) {numberoftimesteps=val;}
    inline void push_back(OneTimeStepT<T> val){Qs.push_back(val);}
    inline void SetOperatormap(const unsigned long long map) 
    {operatormap=map;}
    void resample(const double &dt,
		  const unsigned long long &maxstep=0);
    unsigned int CopyToBuffer(unsigned int &bufferpos,char *buffer);
    unsigned int ReadFromBuffer(unsigned int &bufferpos,char *buffer);
    std::vector<double> DoTrapezoid(int k,int l, int M, 
				    std::vector<double> *traps=0);
    std::vector<std::pair<bool,double> > 
    DoCFD(int delay, double Amplification,
	  double threshold, std::vector<double> *cfss=0);
    void AddNoise(const double &RMS);
    bool DoTAlign(const double &Amp,const unsigned int target);
    void PatchLeft(const unsigned int numberofsmps);
    void PadRight(const unsigned int numberofsmps);
    void TruncateAt(const unsigned int numberofsmps);
    bool TimeShift(const double &dt/*s*/);
    int GetHitSegment();
    int GetNbNetChargeSegments(double threshold=1000);
    int GetNbNonZeroSegments(double threshold=1000);
    void CopyFlatToVector(std::vector<double> &fVec);
  private:
    std::vector<OneTimeStepT<T> > Qs;
    float x,y,z;
    std::vector<short> qsatt;
    unsigned int numberoftimesteps;
    unsigned long int operatormap;
  };

  /**
     streamer to write a pulseshape to a stream
  */
  template <typename T>
  std::ostream& operator << (std::ostream& os, 
			     AGATAGeFEM::APulseShapeT<T> &apulse);
  /**
     streamer to read a pulseshape from a stream
  */
  template <typename T>
  std::istream& operator >> (std::istream& os, 
			     AGATAGeFEM::APulseShapeT<T> &apulse);
  /**
     Function to sum-up pulseshapes with weights and scale of final amplitude 
     "result" should be an empty pulse
  */
  template <typename T1,typename T2>
  void SumPulseShapes(std::vector<std::pair<APulseShapeT<T1>,double> > &pulses,
		      const double &amplitude,APulseShapeT<T2> &result,
		      bool UseT=false);


  typedef APulseShapeT<short> APulseShape;
#endif

#ifndef _APulesShapeEvent_
#define _APulesShapeEvent_
  /**
     A class to hold a pulseshape plus an event nb and timestamp
  */
  template <typename T>
  class APulseShapeEvent {
  public:
    APulseShapeEvent(){thepulseshape = new APulseShapeT<T>;}
    ~APulseShapeEvent(){delete thepulseshape;}
    APulseShapeT<T> *operator()(){return thepulseshape;}
    void SetEvtNb(unsigned long int val){evtnb=val;}
    void SetTimeStamp(double val){timestamp=val;}
    unsigned long int GetEvtNb(){return evtnb;}
    double GetTimeStamp(){return timestamp;}
  private:
    APulseShapeT<T> *thepulseshape;
    unsigned long int evtnb;
    double timestamp;
  };
  /**
     Streamers for APulseShapeEvent
     << output
  */
  //  template <typename T>
  //  std::ostream& operator << (std::ostream& os, 
  //			     AGATAGeFEM::APulseShapeEvent<T> &anevent);
  template <typename T>
  std::ostream& operator << (std::ostream& os, 
			     AGATAGeFEM::APulseShapeEvent<T> &anevent)
    
  {
    os << (*anevent());
    unsigned long int evn = anevent.GetEvtNb();
    double timestamp = anevent.GetTimeStamp();
    os.write((char*)&evn,sizeof(unsigned long int));
    os.write((char*)&timestamp,sizeof(double));
    return os;
  }

  /**
     Streamers for APulseShapeEvent
     >> input
  */
  template <typename T>
  std::istream& operator >> (std::istream& os, 
			     AGATAGeFEM::APulseShapeEvent<T> &anevent)
    
  {
    os >> (*anevent());
    unsigned long int evn;
    double timestamp;
    os.read((char*)&evn,sizeof(unsigned long int));
    os.read((char*)&timestamp,sizeof(double));
    anevent.SetTimeStamp(timestamp);
    anevent.SetEvtNb(evn);
    return os;
  }
#endif



  using namespace AGATAGeFEM;


  template <typename T>
  std::ostream& AGATAGeFEM::operator << (std::ostream& os, 
					 AGATAGeFEM::APulseShapeT<T> &apulse) 
  {return apulse.write(os);}

  template 
  std::ostream& AGATAGeFEM::operator <<
  (std::ostream& os, AGATAGeFEM::APulseShapeT<short> &apulse);

  template <typename T>
  std::istream& AGATAGeFEM::operator >> (std::istream& os, 
					 AGATAGeFEM::APulseShapeT<T> &apulse)
  {return apulse.read(os);}

  template 
  std::istream& AGATAGeFEM::operator >>
  (std::istream& os, AGATAGeFEM::APulseShapeT<short> &apulse);

  template <typename T>
  std::ostream& APulseShapeT<T>::write(std::ostream& os)

  {
    unsigned short endian = 1;
    numberoftimesteps = Qs.size();
    os.write((char*)&EventStartToken,8);
    os.write((char*)&endian,sizeof(unsigned short));
    os.write((char*)&x,sizeof(float));
    os.write((char*)&y,sizeof(float));
    os.write((char*)&z,sizeof(float));
    os.write((char*)&numberoftimesteps,sizeof(unsigned int));
    for (int atsmp=0; atsmp<numberoftimesteps; atsmp++){
      os.write((char*)&Qs[atsmp].t,sizeof(double));  
      for (int atsegment=0; atsegment<37; atsegment++){
	os.write((char*)&Qs[atsmp].Q[atsegment],sizeof(T));
      }
    }
    return os;
  }

  template std::ostream& APulseShapeT<short>::write(std::ostream& os);

  template <typename T>
  std::istream& APulseShapeT<T>::read(std::istream& os)

  {
    unsigned short endian;
    unsigned long int EventStart;
    double d_tmp;
    short s_tmp;
    do{
      os.read((char*)&EventStart,8);    
    }while(EventStart!=AGATAGeFEM::EventStartToken && os.good());
    os.read((char*)&endian,sizeof(unsigned short));
    if(endian==1){
      os.read((char*)&x,sizeof(float));
      os.read((char*)&y,sizeof(float));
      os.read((char*)&z,sizeof(float));
      os.read((char*)&numberoftimesteps,sizeof(unsigned int));
      Qs.resize(numberoftimesteps);
      for (int atsmp=0; atsmp<numberoftimesteps; atsmp++){
	os.read((char*)&d_tmp,sizeof(double));
	Qs[atsmp].t = d_tmp;
	for (int atsegment=0; atsegment<37; atsegment++){
	  os.read((char*)&s_tmp,sizeof(T));
	  Qs[atsmp].Q[atsegment]=s_tmp;
	}
      }
    }
    return os;
  }


  template std::istream& APulseShapeT<short>::read(std::istream& os);
  template std::istream& APulseShapeT<double>::read(std::istream& os);

  template <typename T>
  unsigned int APulseShapeT<T>::CopyToBuffer(unsigned int &bufferpos,
					     char *buffer)

  {
    unsigned int offset = bufferpos;
    unsigned short endian = 1;
    numberoftimesteps = Qs.size();
    memcpy(buffer+bufferpos,(void*)&EventStartToken,8);
    bufferpos+=8;
    memcpy(buffer+bufferpos,(void*)&endian,sizeof(unsigned short));
    bufferpos+=sizeof(unsigned short);
    memcpy(buffer+bufferpos,(void*)&x,sizeof(float));
    bufferpos+=sizeof(float);
    memcpy(buffer+bufferpos,(void*)&y,sizeof(float));
    bufferpos+=sizeof(float);
    memcpy(buffer+bufferpos,(void*)&z,sizeof(float));
    bufferpos+=sizeof(float);
    memcpy(buffer+bufferpos,(void*)&numberoftimesteps,sizeof(unsigned int));
    bufferpos+=sizeof(unsigned int);
    for (int atsmp=0; atsmp<numberoftimesteps; atsmp++){
      memcpy(buffer+bufferpos,(void*)&Qs[atsmp].t,sizeof(double));  
      bufferpos+=sizeof(double);
      for (int atsegment=0; atsegment<37; atsegment++){
	memcpy(buffer+bufferpos,(void*)&Qs[atsmp].Q[atsegment],sizeof(T));
	bufferpos+=sizeof(T);
      }
    }
    return bufferpos-offset;
  }

  template unsigned int APulseShapeT<short>::
  CopyToBuffer(unsigned int &bufferpos,char *buffer);
  template unsigned int APulseShapeT<double>::
  CopyToBuffer(unsigned int &bufferpos,char *buffer);

  template <typename T>
  unsigned int APulseShapeT<T>::ReadFromBuffer(unsigned int &bufferpos,
					       char *buffer)
  {
    unsigned int offset = bufferpos;
    unsigned long int EventStart;
    double d_tmp;
    short s_tmp;
    unsigned short endian;
    memcpy((char*)&EventStart,buffer+bufferpos,8);
    bufferpos+=8;
    if(EventStart!=AGATAGeFEM::EventStartToken){
      std::clog << "Bad buffer alignment|||"
		<< __LINE__ << " " << __FILE__ << std::endl;
    }
    memcpy((void*)&endian,buffer+bufferpos,sizeof(unsigned short));
    bufferpos+=sizeof(unsigned short);
    memcpy((void*)&x,buffer+bufferpos,sizeof(float));
    bufferpos+=sizeof(float);
    memcpy((void*)&y,buffer+bufferpos,sizeof(float));
    bufferpos+=sizeof(float);
    memcpy((void*)&z,buffer+bufferpos,sizeof(float));
    bufferpos+=sizeof(float);
    memcpy((void*)&numberoftimesteps,buffer+bufferpos,sizeof(unsigned int));
    bufferpos+=sizeof(unsigned int);
    Qs.resize(numberoftimesteps);
    for (int atsmp=0; atsmp<numberoftimesteps; atsmp++){
      memcpy((void*)&Qs[atsmp].t,buffer+bufferpos,sizeof(double));  
      bufferpos+=sizeof(double);
      for (int atsegment=0; atsegment<37; atsegment++){
	memcpy((void*)&Qs[atsmp].Q[atsegment],buffer+bufferpos,sizeof(T));
	bufferpos+=sizeof(T);
      }
    }
    return bufferpos-offset;
  }

  template unsigned int APulseShapeT<short>::
  ReadFromBuffer(unsigned int &bufferpos,char *buffer);
  template unsigned int APulseShapeT<double>::
  ReadFromBuffer(unsigned int &bufferpos,char *buffer);


  template <typename T> template <typename T2>
  APulseShapeT<T>& APulseShapeT<T>::operator=(const APulseShapeT<T2>& rhs)

  {
    if((void*)this!=(void*)&rhs){//Check this is not a=a
      //Check size of container
      if(Qs.size()!=rhs.size()) Qs.resize(rhs.size());
      //copy contents
      x = rhs.getx();
      y = rhs.gety();
      z = rhs.getz();
      numberoftimesteps = rhs.getnumberoftimesteps();
      for(unsigned int i=0; i<rhs.size(); ++i){
	Qs[i].t=rhs[i].t;
	for(unsigned int j=0; j<37; ++j) Qs[i].Q[j] = rhs[i].Q[j];
      }
      return *this;
    } else  return *this;
  }

  template APulseShapeT<short>& 
  APulseShapeT<short>::operator=(const APulseShapeT<double>& rhs);

  template <typename T>
  APulseShapeT<T>& APulseShapeT<T>::operator+=(const APulseShapeT<T>& rhs)

  {
    for(unsigned int i=0; i<rhs.size(); ++i){
      for(unsigned int j=0; j<37; ++j) Qs[i].Q[j] += rhs[i].Q[j];
    }
    return *this;
  }

  template <typename T>
  APulseShapeT<T>& APulseShapeT<T>::operator-=(const APulseShapeT<T>& rhs)

  {
    for(unsigned int i=0; i<rhs.size(); ++i){
      for(unsigned int j=0; j<37; ++j) Qs[i].Q[j] -= rhs[i].Q[j];
    }
    return *this;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator+(const APulseShapeT<T>& rhs)

  {
    return APulseShapeT<T>(*this)+=rhs;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator-(const APulseShapeT<T>& rhs)

  {
    return APulseShapeT<T>(*this)-=rhs;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator*(const double& rhs)

  {
    APulseShapeT<T> ap(*this);
    for(unsigned int i=0; i<Qs.size(); ++i){
      for(unsigned int j=0; j<37; ++j){
	ap[i].Q[j]*=rhs;
      }
    }
    return ap;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator/(const double& rhs)

  {
    APulseShapeT<T> ap(*this);
    for(unsigned int i=0; i<Qs.size(); ++i){
      for(unsigned int j=0; j<37; ++j){
	ap[i].Q[j]/=rhs;
      }
    }
    return ap;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator*(const float& rhs)

  {
    APulseShapeT<T> ap(*this);
    for(unsigned int i=0; i<Qs.size(); ++i){
      for(unsigned int j=0; j<37; ++j){
	ap[i].Q[j]*=rhs;
      }
    }
    return ap;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator/(const float& rhs)

  {
    APulseShapeT<T> ap(*this);
    for(unsigned int i=0; i<Qs.size(); ++i){
      for(unsigned int j=0; j<37; ++j){
	ap[i].Q[j]/=rhs;
      }
    }
    return ap;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator*(const int& rhs)

  {
    APulseShapeT<T> ap(*this);
    for(unsigned int i=0; i<Qs.size(); ++i){
      for(unsigned int j=0; j<37; ++j){
	ap[i].Q[j]*=rhs;
      }
    }
    return ap;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator/(const int& rhs)

  {
    APulseShapeT<T> ap(*this);
    for(unsigned int i=0; i<Qs.size(); ++i){
      for(unsigned int j=0; j<37; ++j){
	ap[i].Q[j]/=rhs;
      }
    }
    return ap;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator*(const short& rhs)

  {
    APulseShapeT<T> ap(*this);
    for(unsigned int i=0; i<Qs.size(); ++i){
      for(unsigned int j=0; j<37; ++j){
	ap[i].Q[j]*=rhs;
      }
    }
    return ap;
  }

  template <typename T>
  const APulseShapeT<T> APulseShapeT<T>::operator/(const short& rhs)

  {
    APulseShapeT<T> ap(*this);
    for(unsigned int i=0; i<Qs.size(); ++i){
      for(unsigned int j=0; j<37; ++j){
	ap[i].Q[j]/=rhs;
      }
    }
    return ap;
  }

  template APulseShapeT<short>& 
  APulseShapeT<short>::operator=(const APulseShapeT<short>& rhs);
  template APulseShapeT<short>& 
  APulseShapeT<short>::operator+=(const APulseShapeT<short>& rhs);
  template APulseShapeT<short>& 
  APulseShapeT<short>::operator-=(const APulseShapeT<short>& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator+(const APulseShapeT<short>& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator-(const APulseShapeT<short>& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator*(const double& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator/(const double& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator*(const float& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator/(const float& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator*(const int& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator/(const int& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator*(const short& rhs);
  template const APulseShapeT<short> 
  APulseShapeT<short>::operator/(const short& rhs);

  template <typename T>
  std::vector<double> APulseShapeT<T>::operator()(double t,int *ll)

  {
    std::vector<double> res;
    res.resize(37);
    //First check for trivial cases
    if(t<0){
      return res;
    }
    if(t>Qs[Qs.size()-1].t){
      for(int atseg=0; atseg<37; atseg++){
	res[atseg]=Qs[Qs.size()-1].Q[atseg];
      }
      return res;
    }
    int li=0,hi=Qs.size()-1;
    if(ll){//Are we caching the lower index?
      hi=*ll;
      while(Qs[hi].t<t && hi<Qs.size()-1) ++hi;
      li = hi-1;
      if(ll) *ll=li;
    } else {
      //if not, to find the good pair of times, we bisect
      do{
	int index=(hi+li)>>1;//integer division by 2
	double ttmp = Qs[index].t;
	if(ttmp>t) hi=index; else li=index;
      }while(abs(li-hi)>1);
    }
    double tl = Qs[li].t;
    double th = Qs[hi].t;
    double dt = th-tl;
    double tstep = (t-tl);
    double tstep_dt = tstep/dt;
    unsigned long int mask=1;//We can mask off signals we don't use
    for(int atseg=0; atseg<37; atseg++){
      if(operatormap&mask){
	double ql = Qs[li].Q[atseg];
	double qh = Qs[hi].Q[atseg];
	double dQ = qh-ql;
	res[atseg] = ql+dQ*tstep_dt;
      } else res[atseg]=0;
      mask=mask<<1;
    }
    return res;
  }

  template <typename T> bool APulseShapeT<T>::TimeShift(const double &dt/*s*/)

  {
    if(dt>0){//This operation makes since
      double tstop=Qs.back().t;
      //First we add the dt to all the times
      for(unsigned int atsmp=0; atsmp<Qs.size(); ++atsmp){
	Qs[atsmp].t+=dt;
      }
      //The we pad a zero before...
      OneTimeStepT<T> astep;
      astep.t=0;
      std::fill(astep.Q,astep.Q+37,0);
      Qs.insert(Qs.begin(),astep);
      //And get rid of all points later than tstop
      while(Qs.back().t>tstop) Qs.pop_back();
      //And add one point at tstop
      astep.t=tstop;
      for(int ate=0; ate<37; ate++){
	astep.Q[ate]=Qs.back().Q[ate];
      }
      Qs.push_back(astep);
      numberoftimesteps=Qs.size();
      return true;
    } else {
      //We do nothing but return false
      return false;
    }
  }


  template bool APulseShapeT<double>::TimeShift(const double &dt);
  template bool APulseShapeT<short>::TimeShift(const double &dt);
  /*
    std::vector<short> & APulseShape::operator()(const double &t)//Fast version

    {
    if(qsatt.size()!=37) qsatt.resize(37);
    //First check for trivial cases
    if(t<0){
    for(int atseg=0; atseg<37; atseg++){
    qsatt[atseg]=0;
    }
    return qsatt;
    }
    if(t>Qs[Qs.size()-1].t){
    for(int atseg=0; atseg<37; atseg++){
    qsatt[atseg]=Qs[Qs.size()-1].Q[atseg];
    }
    return qsatt;
    }
    //Find the good pair of times, we bisect
    int li=0,hi=Qs.size()-1;
    do{
    int index=(hi+li)>>1;//integer division by 2
    double ttmp = Qs[index].t;
    if(ttmp>t) hi=index; else li=index;
    }while(abs(li-hi)>1);
    double tl = Qs[li].t;
    double th = Qs[hi].t;
    double tstep_dt = (t-tl)/(th-tl);
    unsigned long int mask=1;//We can mask off signals we don't use
    for(int atseg=0; atseg<37; atseg++){
    if(operatormap&mask){
    short ql = Qs[li].Q[atseg];
    short qh = Qs[hi].Q[atseg];
    short dQ = qh-ql;
    qsatt[atseg] = ql+dQ*tstep_dt;
    } else qsatt[atseg]=0;
    mask=mask<<1;
    }
    return qsatt;
    }
  */


  template std::vector<double> APulseShapeT<short>::operator()(double t,int *ll);
  template std::vector<double> APulseShapeT<double>::operator()(double t,
								int *ll);

  template <typename T>
  void APulseShapeT<T>::resample(const double &dt,
				 const unsigned long long &maxstep)

  {
    std::vector<OneTimeStepT<T> > qtmp;
    double tmax = Qs.back().t;
    OneTimeStepT<T> astep;
    astep.t=0;
    for(int i=0; i<37; i++) astep.Q[i]=0;
    qtmp.push_back(astep);
    double t=dt;
    int atpos=0;
    while(t<tmax+dt/2){
      while(Qs[atpos].t<t && atpos<Qs.size()-1) atpos++;
      double tl = Qs[atpos-1].t;
      double th = Qs[atpos].t;
      double tstep_dt = (t-tl)/(th-tl);
      for(int atseg=0; atseg<37; atseg++){
	T ql = Qs[atpos-1].Q[atseg];
	T qh = Qs[atpos].Q[atseg];
	T dQ = qh-ql;
	astep.Q[atseg] = ql+dQ*tstep_dt;
      }
      astep.t=t;
      qtmp.push_back(astep);
      if(maxstep){
	if(qtmp.size()==maxstep) break;
      }
      t+=dt;
    } 
    Qs=qtmp;
    numberoftimesteps=qtmp.size();
  }

  template void APulseShapeT<short>::resample(const double &dt,
					      const unsigned long long &maxstep);
  template void APulseShapeT<double>::resample(const double &dt,
					       const unsigned long long &maxstep);



  template <typename T>
  std::vector<double> APulseShapeT<T>::DoTrapezoid(int k,int l, int M, 
						   std::vector<double> *traps)

  {
    std::vector<double> theEs;
    theEs.resize(37);
    if(traps) traps->clear();
    double alpha = M>0 ? 1-exp(-1./M) : 0;
    for(int seg=0; seg<37; seg++){
      double avgE=0, d = 0, p = 0, s=0;
      for (int i=0; i<numberoftimesteps; i++){
	/*Trap. Shaping*/
	if(i-1>=0) d+=Qs[i-1].Q[seg];
	if(i-k-l-1>=0) d-=Qs[i-k-l-1].Q[seg];
	s=Qs[i].Q[seg];
	if(i-k-l>0) s-=Qs[i-k-l].Q[seg];
	s+=alpha*d;
	if(i>k-l/2 && i<=k+l/2) theEs[seg]+=s;
	if(traps) (*traps).push_back(s);
      }
      theEs[seg]/=l;
    }
    return theEs;
  }

  template
  std::vector<double> APulseShapeT<double>::DoTrapezoid(int k,int l, int M, 
							std::vector<double> *traps);
  template 
  std::vector<double> APulseShapeT<short>::DoTrapezoid(int k,int l, int M, 
						       std::vector<double> *traps);


  template <typename T>
  std::vector<std::pair<bool,double> > 
  APulseShapeT<T>::DoCFD(int delay, double Amplification,
			 double threshold, 
			 std::vector<double> *cfds)
  
  {
    std::vector<std::pair<bool,double> > thecfds;
    thecfds.resize(37);
    if(cfds) cfds->clear();
    std::vector<double> cfdtrace;
    cfdtrace.resize(numberoftimesteps);
    double coreend = Qs[numberoftimesteps-1].Q[0];
    for(int seg=0; seg<37; seg++){
      bool above=false;
      for (int smp=0; smp<numberoftimesteps; smp++){
	if(fabs(Qs[smp].Q[seg])>threshold*fabs(coreend)) above=true;
	cfdtrace[smp]=Qs[smp].Q[seg] - 
	  (smp>delay ? Qs[smp-delay].Q[seg]*Amplification : 0);
	if(cfds) (*cfds).push_back(cfdtrace[smp]);
      }
      if(above){
	//First find max and min
	std::vector<double>::iterator startit,maxit = 
	  std::max_element(cfdtrace.begin(),cfdtrace.end());
	std::vector<double>::iterator stopit,minit = 
	  std::min_element(cfdtrace.begin(),cfdtrace.end());
	if(maxit==minit || 
	   (maxit == cfdtrace.end() || minit == cfdtrace.end())) {
	  thecfds[seg]=std::pair<bool,double>(false,0.);
	} else {
	  startit = (maxit-cfdtrace.begin())<(minit-cfdtrace.begin()) 
					     ? maxit : minit;
	  stopit = startit;
	  ++stopit;
	  while(*stopit**startit>0 && stopit!=cfdtrace.end()) {
	    ++startit;++stopit;
	  }
	  //stopit after zero crossing, startit before zero crossing
	  double time = (startit-cfdtrace.begin())
	    - *startit/(*stopit-*startit);
	  thecfds[seg]=std::pair<bool,double>(true,time);
	}
      } else {
	thecfds[seg]=std::pair<bool,double>(false,0.);
      }
    }
    return thecfds;
  }

  template std::vector<std::pair<bool,double> >
  APulseShapeT<double>::DoCFD(int delay, 
			      double Amplification,
			      double threshold, 
			      std::vector<double> *cfds);
  template std::vector<std::pair<bool,double> > 
  APulseShapeT<short>::DoCFD(int delay, 
			     double Amplification,
			     double threshold, 
			     std::vector<double> *cfds);



  template <typename T>
  void APulseShapeT<T>::AddNoise(const double &RMS)

  {
    for(unsigned int smp=0; smp<Qs.size(); smp++){
      for(int seg=0; seg<37; ++seg){
	Qs[smp].Q[seg]+=0;
      }
    }
  }


  template void APulseShapeT<double>::AddNoise(const double &RMS);
  template void APulseShapeT<short>::AddNoise(const double &RMS);


  template <typename T>
  void APulseShapeT<T>::PatchLeft(const unsigned int numberofsmps)

  {
    OneTimeStepT<T> astep;
    for(int i=0; i<37; i++) astep.Q[i]=0;
    double dt = Qs[1].t-Qs[0].t;
    for(unsigned int i=0; i<numberofsmps; i++){
      Qs.insert(Qs.begin(),astep);
    }
    for(unsigned int i=0; i<Qs.size(); ++i){
      Qs[i].t=dt*i;
    }
    numberoftimesteps=Qs.size();
  }

  template void APulseShapeT<double>::PatchLeft(const unsigned int numberofsmps);
  template void APulseShapeT<short>::PatchLeft(const unsigned int numberofsmps);


  template <typename T>
  void APulseShapeT<T>::PadRight(const unsigned int numberofsmps)

  {
    OneTimeStepT<T> astep = Qs[Qs.size()-1];
    double dt = Qs[1].t-Qs[0].t;
    for(unsigned int i=0; i<numberofsmps; i++){
      Qs.push_back(astep);
    }
    for(unsigned int i=0; i<Qs.size(); ++i){
      Qs[i].t=dt*i;
    }
    numberoftimesteps=Qs.size();
  }

  template void APulseShapeT<double>::PadRight(const unsigned int numberofsmps);
  template void APulseShapeT<short>::PadRight(const unsigned int numberofsmps);

  template <typename T>
  void APulseShapeT<T>::TruncateAt(const unsigned int numberofsmps)

  {
    if(Qs.size()>numberofsmps){
      Qs.resize(numberofsmps);
    }
  }

  template void APulseShapeT<double>::TruncateAt(const unsigned int numberofsmps);
  template void APulseShapeT<short>::TruncateAt(const unsigned int numberofsmps);


  template <typename T>
  bool APulseShapeT<T>::DoTAlign(const double &Amp, 
				 const unsigned int target)

  {
    unsigned int atsmp=1;
    do{
      if(Qs[atsmp].Q[0]>Amp){//We crossed LE trigger
	//Find "exact" crossing
	double dy = Qs[atsmp].Q[0]-Qs[atsmp-1].Q[0];
	double dt = Qs[atsmp].t-Qs[atsmp-1].t;
	/*
	  y = m+k*x
	  x = (y-m)/k
	*/
	double dsmp = (Amp-Qs[atsmp-1].Q[0])/dy;
	//Shift all y values to the "left" with (smp-1+dsmp)-target
	std::vector<OneTimeStepT<T> > Qstmp;
	Qstmp.resize(Qs.size());
	int ll=0;
	for(unsigned int i=0; i<Qs.size(); i++){
	  double t = Qs[i].t+((atsmp-1+dsmp)-target)*dt;
	  std::vector<double> Q=(*this)(t,&ll);
	  for(int seg=0; seg<37; seg++){
	    Qstmp[i].Q[seg]=Q[seg];
	  }
	  Qstmp[i].t=Qs[i].t;
	}
	Qs=Qstmp;
	return 1;
      }
    }while(++atsmp<Qs.size());
    return 0;
  }



  template bool APulseShapeT<double>::DoTAlign(const double &Amp,
					       const unsigned int target);
  template bool APulseShapeT<short>::DoTAlign(const double &Amp,
					      const unsigned int target);


  template  <typename T> int APulseShapeT<T>::GetHitSegment()

  {
    OneTimeStepT<T> lts = Qs[Qs.size()-1];
    auto it1 = std::begin(lts.Q);
    ++it1;
    auto it2 = std::end(lts.Q);
    //  return
    //  std::distance(lts.Q+1,std::min_element(lts.Q+1,lts.Q+37));
    return std::distance(it1,std::min_element(it1,it2));
  }

  template int APulseShapeT<double>::GetHitSegment();
  template int APulseShapeT<short>::GetHitSegment();


  template  <typename T> int
  APulseShapeT<T>::GetNbNetChargeSegments(double threshold)

  {
    OneTimeStepT<T> lts = Qs[Qs.size()-1];
    int nbhitseg=0;
    auto it1 = std::begin(lts.Q);
    ++it1;
    auto it2 = std::end(lts.Q);
    for(; it1!=it2; ++it1){
      if(*it1<-threshold) nbhitseg++;
    }
    return nbhitseg;
  }

  template int APulseShapeT<double>::GetNbNetChargeSegments(double threshold);
  template int APulseShapeT<short>::GetNbNetChargeSegments(double threshold);


  template  <typename T> int
  APulseShapeT<T>::GetNbNonZeroSegments(double threshold)

  {
    OneTimeStepT<T> lts = Qs[Qs.size()-1];
    int nbnonzeroseg=0;
    auto it1 = std::begin(lts.Q);
    ++it1;
    auto it2 = std::end(lts.Q);
    for(; it1!=it2; ++it1){
      if(fabs(*it1)>threshold) nbnonzeroseg++;
    }
    return nbnonzeroseg;
  }

  template int APulseShapeT<double>::GetNbNonZeroSegments(double threshold);
  template int APulseShapeT<short>::GetNbNonZeroSegments(double threshold);



  template  <typename T>
  void APulseShapeT<T>::CopyFlatToVector(std::vector<double> &fVec)

  {
    for(int atseg=0; atseg<sizeof(Qs[0].Q)/sizeof(Qs[0].Q[0]); atseg++){
      for(int att=0; att<numberoftimesteps; att++){
	fVec.push_back(Qs[att].Q[atseg]);
      }
    }
  }

  template void APulseShapeT<short>::CopyFlatToVector(std::vector<double> &fVec);
  template void APulseShapeT<double>::
  CopyFlatToVector(std::vector<double> &fVec);

  template <typename T1, typename T2>
  void AGATAGeFEM::SumPulseShapes(std::vector<std::pair<APulseShapeT<T1>,
				  double> > &pulses,const double &amplitude,
				  APulseShapeT<T2> &result,bool UseT)
  {
    typename std::vector<std::pair<APulseShapeT<T1> ,double> >::iterator itp = 
      pulses.begin();
    double Qts[37],t;
    OneTimeStepT<T2> Q;
    for(int ts=0; ts<pulses[0].first.getnumberoftimesteps(); ts++){
      for(int i=0; i<37; i++) Qts[i]=0;
      t = pulses[0].first[ts].t;
      for(itp = pulses.begin();itp!=pulses.end();++itp){
	std::vector<double> Qft;
	if(UseT){
	  Qft=itp->first(t);
	}
	for(int i=0; i<37; i++){
	  if(!UseT) Qts[i] += itp->first[ts].Q[i]*itp->second;
	  else Qts[i] += Qft[i]*itp->second;
	}
      }
      for(int i=0; i<37; i++) Q.Q[i] = Qts[i]*amplitude;
      Q.t = t;
      result.push_back(Q);
    }
  }

}
template void 
AGATAGeFEM::SumPulseShapes(std::vector<std::pair<APulseShapeT<short>, double> >
			   &pulses,const double &amplitude,
			   APulseShapeT<short> &result,bool UseT);
template void 
AGATAGeFEM::SumPulseShapes(std::vector<std::pair<APulseShapeT<double>, double> >
			   &pulses,const double &amplitude,
			   APulseShapeT<double> &result,bool UseT);





int main(int argc, char** argv)

{
  std::ifstream input(argv[1]);
  std::ofstream ofs ("test.csv", std::ofstream::out);
  AGATAGeFEM::APulseShapeEvent<short> AnPSEvent;
  while(input >> AnPSEvent){
    if(input.good()){
      ofs << -1 << "," 
		<< std::setprecision(5) << AnPSEvent.GetTimeStamp()
		<< std::endl;
      AnPSEvent()->resample(10e-9);
      for(int i=0; i<AnPSEvent()->getnumberoftimesteps(); i++){
	ofs << ","  << (*AnPSEvent())[i].t ;
	for(int seg=0; seg<37; seg++){
	  ofs << "," << (*AnPSEvent())[i].Q[seg] ;
	}
	ofs << std::endl;
      }
    }
  }
  return 0;
}
