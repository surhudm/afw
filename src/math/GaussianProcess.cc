// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/**
 * @file GaussianProcess.cc
 *
 * @ingroup afw
 *
 * @author Scott Daniel
 * Contact: scott.f.daniel@gmail.com
*/

#include "lsst/afw/math/GaussianProcess.h"
//#include "gptest/gptest.h"
#include "lsst/afw/math/detail/GaussianProcessFunctions.h"
#include <iostream>
#include <cmath>

using namespace std;


namespace lsst{
namespace afw{
namespace math {

namespace GPfn = lsst::afw::math::detail::gaussianProcess;

GaussianProcessTimer::~GaussianProcessTimer(){};

GaussianProcessTimer::GaussianProcessTimer(){
    interpolationCount=0;
    iterationTime=0.0;
    eigenTime=0.0;
    searchTime=0.0;
    varianceTime=0.0;
    totalTime=0.0;

}

void GaussianProcessTimer::reset(){
    interpolationCount=0;
    iterationTime=0.0;
    eigenTime=0.0;
    searchTime=0.0;
    varianceTime=0.0;
    totalTime=0.0;
}

template <typename T>
KdTree<T>::~KdTree(){}

template <typename T>
KdTree<T>::KdTree(ndarray::Array<T,2,2> const &dt)
{
  

  int i;
  
  _pts=dt.template getSize<0>();
  _dimensions=dt.template getSize<1>();
  
  //buffers to use when first building the tree
  _toSort=allocate(ndarray::makeVector(_pts));
  _inn=allocate(ndarray::makeVector(_pts));

  _roomStep=5000;
  _room=_pts;
  
  data=allocate(ndarray::makeVector(_room,_dimensions));
  
  data.deep()=dt;
  
  _tree=allocate(ndarray::makeVector(_room,4));
 
  for(i=0;i<_pts;i++){
    _inn[i]=i;
  }
  
  _organize(_inn,_pts,-1,-1);
  
}

template<typename T>
void KdTree<T>::findNeighbors(ndarray::Array<int,1,1> neighdex,
                              ndarray::Array<double,1,1> dd,
                              ndarray::Array<const T,1,1> const &v,
                              int n_nn
                              )
{  
  int i,start;
 
  ndarray::Array<int,1,1> order;
  ndarray::Array<double,1,1> dorder;  
  
  order=allocate(ndarray::makeVector(3));
  dorder=allocate(ndarray::makeVector(3));
  
  _neighborCandidates=allocate(ndarray::makeVector(n_nn));
  _neighborDistances=allocate(ndarray::makeVector(n_nn));
  _neighborsFound=0;
  _neighborsWanted=n_nn;
  
  for(i=0;i<n_nn;i++)_neighborDistances[i]=-1.0;
  
  start=_findNode(v);

  _neighborDistances[0]=_distance(v,data[start]);
  _neighborCandidates[0]=start;
  _neighborsFound=1;
 
  
  if(_tree[start][3]>=0){
    dorder[2]=_distance(v,data[_tree[start][3]]);
  
  }
  else dorder[2]=-1.0;
  order[2]=3;
  
  if(_tree[start][1]>=0){
    dorder[0]=_distance(v,data[_tree[start][1]]);
  
  }
  else dorder[0]=-1.0;
  order[0]=1;
  
  if(_tree[start][2]>=0){
    dorder[1]=_distance(v,data[_tree[start][2]]);
  
  }
  else dorder[1]=-1.0;
  order[1]=2;
  
  GPfn::mergeSort<double>(dorder,order,3);
  
  //search the branches in ascending order of distance from the test point
  //the idea being that if we look at branches that are closer first, we will
  //be more likely to rule out points quicker, speeding the search
  for(i=0;i<3;i++){
    if(_tree[start][order[i]]>=0){
      _lookForNeighbors(v,_tree[start][order[i]],start);
    }
  }
  
  for(i=0;i<n_nn;i++){
    neighdex[i]=_neighborCandidates[i];
    dd[i]=_neighborDistances[i];
  }
  
 
  
}

template <typename T>
void KdTree<T>::addPoint(ndarray::Array<const T,1,1> const &v){

  int i,j,node,dim;
  
  node=_findNode(v);
  dim=_tree[node][0]+1;
  if(dim==_dimensions)dim=0;
  
  if(_pts==_room){
    
    ndarray::Array<T,2,2> dbuff=allocate(ndarray::makeVector(_pts,_dimensions));
  
    
    ndarray::Array<int,2,2> tbuff=allocate(ndarray::makeVector(_pts,4));
  
   
   dbuff.deep()=data;
   tbuff.deep()=_tree;

    _room+=_roomStep;
   
    _tree=allocate(ndarray::makeVector(_room,4)); 
    data=allocate(ndarray::makeVector(_room,_dimensions));
    
    
    for(i=0;i<_pts;i++){
      for(j=0;j<_dimensions;j++)data[i][j]=dbuff[i][j];
      for(j=0;j<4;j++)_tree[i][j]=tbuff[i][j];
   
    }

  }
  
  _tree[_pts][0]=dim;
  _tree[_pts][3]=node;
  i=_tree[node][0];
  
  if(data[node][i]>v[i]){
    if(_tree[node][1]>=0){
      std::cout<<"WARNING adding to a piece of tree that already exists 1\n";
      std::cout<<"node "<<node<<" "<<_tree[node][1]<<" "<<data[node][i]<<" "<<v[i]<<"\n";
      std::cout<<"pts "<<_pts<<"\n";
    }
    _tree[node][1]=_pts;
  }
  else{
    if(_tree[node][2]>=0){
      std::cout<<"WARNING adding to a piece of tree that already exists 2\n";
      std::cout<<"node "<<node<<" "<<_tree[node][2]<<" "<<data[node][i]<<" "<<v[i]<<"\n"; 
      std::cout<<"pts "<<_pts<<"\n";
    }
    _tree[node][2]=_pts;
  }
  _tree[_pts][1]=-1;
  _tree[_pts][2]=-1;
  for(i=0;i<_dimensions;i++){
    data[_pts][i]=v[i];
  }
  
  _pts++;
  
}

template <typename T>
int KdTree<T>::getPoints(){
  return _pts;
}

template <typename T>
void KdTree<T>::getTreeNode(ndarray::Array<int,1,1> const &v, int dex){
  v[0]=_tree[dex][0];
  v[1]=_tree[dex][1];
  v[2]=_tree[dex][2];
  v[3]=_tree[dex][3];
}

template <typename T>
int KdTree<T>::testTree(){

  int i,j,*isparent,output;
  
  j=0;
  for(i=0;i<_pts;i++){
    if(_tree[i][3]<0)j++;
  }
  if(j!=1){
    std::cout<<"_tree FAILURE "<<j<<" _masterParents\n";
    return 0;
  }
  
  isparent=new int[_pts];
  for(i=0;i<_pts;i++)isparent[i]=0;
  for(i=0;i<_pts;i++){
    isparent[_tree[i][3]]++;
  }
  for(i=0;i<_pts;i++){
    if(isparent[i]>2){ 
      std::cout<<"_tree FAILURE "<<i<<" is parent to "<<isparent[i]<<"\n";
      return 0;
    }
  }
  
  delete [] isparent;
  
  for(i=0;i<_pts;i++){
    
    if(_tree[i][3]>=0){
      if(_tree[_tree[i][3]][1]==i)j=1;
      else j=2;

      output=_walkUpTree(_tree[i][3],j,i);
      if(output!=_masterParent)return 0;
    }
  
  }
  //std::cout<<"done with test of KdTree\n";
  if(output!=_masterParent) return 0;
  else return 1;
}

template <typename T>
void KdTree<T>::_organize(ndarray::Array<int,1,1> const &use, 
                          int ct, 
                          int parent, 
                          int dir
                          )
{
    
  int i,j,k,l,idim,daughter;
  T mean,var,varbest;
   
  if(ct>1){
  //below is code to choose the dimension on which the available points
  //have the greates variance.  This will be the dimension on which
  //the daughter node splits the data
    for(i=0;i<_dimensions;i++){
      mean=0.0;
      var=0.0;
      for(j=0;j<ct;j++){
        mean+=data[use[j]][i];
        var+=data[use[j]][i]*data[use[j]][i];
      }
      mean=mean/double(ct);
      var=var/double(ct)-mean*mean;
      if(i==0 || var>varbest || (var==varbest && parent>=0 && i>_tree[parent][0])){
        idim=i;
        varbest=var;
      }
    
    }//for(i=0;i<_dimensions;i++)
  
    for(i=0;i<ct;i++){
      _toSort[i]=data[use[i]][idim];
    }
  
    GPfn::mergeSort<T>(_toSort,use,ct);
    
    k=ct/2;
    l=ct/2;
    while(k>0 && _toSort[k]==_toSort[k-1])k--;
   
    while(l<ct-1 && _toSort[l]==_toSort[ct/2])l++;
 
    if((ct/2-k)<(l-ct/2) || l==ct-1)j=k;
    else j=l;;
    
    daughter=use[j];

    if(parent>=0)_tree[parent][dir]=daughter;
    _tree[daughter][0]=idim;
    _tree[daughter][3]=parent;

    if(j<ct-1){
      //_organize(&use[j+1],ct-j-1,daughter,2);
      _organize(use[ndarray::view(j+1,use.getSize<0>())],ct-j-1,daughter,2);
    }
    else _tree[daughter][2]=-1;
  
    if(j>0){
      _organize(use,j,daughter,1);
    }
    else _tree[daughter][1]=-1;
    
  }//if(ct>1)
  else{
    daughter=use[0];
    if(parent>=0)_tree[parent][dir]=daughter;
    idim=_tree[parent][0]+1;
    if(idim>=_dimensions)idim=0;
    _tree[daughter][0]=idim;
    _tree[daughter][1]=-1;
    _tree[daughter][2]=-1;
    _tree[daughter][3]=parent;
    
  }
  
  if(parent==-1){
    _masterParent=daughter;
  }
  
}

template <typename T>
int KdTree<T>::_findNode(ndarray::Array<const T,1,1> const &v){
  
  int consider,next,dim;
  
  dim=_tree[_masterParent][0];
  
  if(v[dim]<data[_masterParent][dim])consider=_tree[_masterParent][1];
  else consider=_tree[_masterParent][2];
  
  next=consider;
  
  while(next>=0){
    
    consider=next;
    
    dim=_tree[consider][0];
    if(v[dim]<data[consider][dim])next=_tree[consider][1];
    else next=_tree[consider][2];
  
  }
  
  return consider;
  
}

template<typename T>
void KdTree<T>::_lookForNeighbors(ndarray::Array<const T,1,1> const &v, 
                                  int consider, 
				  int from)
{

  int i,j,going;
  double dd;

  dd=_distance(v,data[consider]);
  
  if(_neighborsFound<_neighborsWanted || dd<_neighborDistances[_neighborsWanted-1]){
    for(j=0;j<_neighborsFound && _neighborDistances[j]<dd;j++);
      
    for(i=_neighborsWanted-1;i>j;i--){
      _neighborDistances[i]=_neighborDistances[i-1];
      _neighborCandidates[i]=_neighborCandidates[i-1];
    }
    
    _neighborDistances[j]=dd;
    _neighborCandidates[j]=consider;
    
    if(_neighborsFound<_neighborsWanted)_neighborsFound++;
  }
  
  if(_tree[consider][3]==from){
    //you came here from the parent
    
    i=_tree[consider][0];
    dd=v[i]-data[consider][i];
    if((dd<=_neighborDistances[_neighborsFound-1] || _neighborsFound<_neighborsWanted) \
    && _tree[consider][1]>=0){
      _lookForNeighbors(v,_tree[consider][1],consider);
    }
    
    dd=data[consider][i]-v[i];
    if((dd<=_neighborDistances[_neighborsFound-1] || _neighborsFound<_neighborsWanted) \
    && _tree[consider][2]>=0){
      _lookForNeighbors(v,_tree[consider][2],consider);
    }
  }
  else{
    //you came here from one of the branches
    
    //descend the other branch
    if(_tree[consider][1]==from){
      going=2;
    }
    else{ 
      going=1;
    }
    
    j=_tree[consider][going];
    
    if(j>=0){
      i=_tree[consider][0];
      if(going==1)dd=v[i]-data[consider][i];
      else dd=data[consider][i]-v[i];
      
      if(dd<=_neighborDistances[_neighborsFound-1] || _neighborsFound<_neighborsWanted){
        _lookForNeighbors(v,j,consider);
      }
    }
    
    //ascend to the parent
    if(_tree[consider][3]>=0){
      
      _lookForNeighbors(v,_tree[consider][3],consider);
      
    }
    
  }
 

}

template <typename T>
int KdTree<T>::_walkUpTree(int target, 
                           int dir, 
			   int root
			   )
{
  //target is the node that you are examining now
  //dir is where you came from
  //root is the ultimate point from which you started
  
  int i,output;
  
  output=1;
  
  if(dir==1){
    if(data[root][_tree[target][0]]>=data[target][_tree[target][0]]){
      std::cout<<"_tree FAILURE root "<<root<<" target "<<target<<" dir "<<dir<<"\n";
      std::cout<<data[root][_tree[target][0]]<<" >= "<<data[target][_tree[target][0]]<<"\n";
      output=0;
      return 0;
      
    }
  }
  else{
      if(data[root][_tree[target][0]]<data[target][_tree[target][0]]){
      
      std::cout<<"_tree FAILURE root "<<root<<"\n";
      std::cout<<" target "<<target<<" dir "<<dir<<" \n";
      std::cout<<data[root][_tree[target][0]]<<" < "<<data[target][_tree[target][0]]<<"\n";
      output=0;
      return 0;

    }
  }
  
  if(_tree[target][3]>=0){
    if(_tree[_tree[target][3]][1]==target)i=1;
    else i=2;
    
    output=output*_walkUpTree(_tree[target][3],i,root);
  
  }
  else{
    output=output*target;
    //so that it will return _masterParent
    //make sure everything is connected to _masterParent
  }
  return output;
  
}

template <typename T>
void KdTree<T>::remove(int target){

  int nl,nr,i,j,k,side;
  int root;
  
  
  nl=0;
  nr=0;
  //printf("about to subtract %d\n",target);
  
  if(_tree[target][1]>=0){
     nl++;
    _count(_tree[target][1],&nl);
  }
  //printf("got nl %d\n",nl);
  
  if(_tree[target][2]>=0){
     nr++;
    _count(_tree[target][2],&nr);
  }
 
  //printf("got nr %d\n",nr);
  
  if(nl==0 && nr==0){

    k=_tree[target][3];
      
      if(_tree[k][1]==target)_tree[k][1]=-1;
      else if(_tree[k][2]==target)_tree[k][2]=-1;
    
  }//if target is terminal
  else if((nl==0 && nr>0) || (nr==0 && nl>0)){
    //printf("lopsided case\n");
    if(nl==0)side=2;
    else side=1;
    
    k=_tree[target][3];
    if(k>=0){//printf("k is non-negative\n");
     if(_tree[k][1]==target){
       _tree[k][1]=_tree[target][side];
       _tree[_tree[k][1]][3]=k;
     }
     else{
       _tree[k][2]=_tree[target][side];
       _tree[_tree[k][2]][3]=k;
     }
    }
    else{
      //printf("ah... the masterparent\n");
      _masterParent=_tree[target][side];
      _tree[_tree[target][side]][3]=-1;
      //printf("assigned the indices\n");
      //printf("continue?");
      //scanf("%d",&i);
    }
    
  
  }//if only one side is populated
  else{
     //printf("hardest case master parent %d\n",masterparent);
     if(nl>nr)side=1;
     else side=2;
     
      k=_tree[target][3];
      if(k<0){
        _masterParent=_tree[target][side];
	_tree[_masterParent][3]=-1;
      }
      else{
        if(_tree[k][1]==target){
         _tree[k][1]=_tree[target][side];
          _tree[_tree[k][1]][3]=k;
        }
        else{
           _tree[k][2]=_tree[target][side];
           _tree[_tree[k][2]][3]=k;
         }
      }
     
     //printf("side is %d\n",side);
     //printf("parent was %d\n",tree[target][3]);
     
     root=_tree[target][3-side];
     
     _descend(root);
     

  }//if both sides are populated
  
    if(target<_pts-1){
      for(i=target+1;i<_pts;i++){
        for(j=0;j<4;j++)_tree[i-1][j]=_tree[i][j];
        for(j=0;j<_dimensions;j++)data[i-1][j]=data[i][j];
     }
    
      for(i=0;i<_pts;i++){
        for(j=1;j<4;j++)if(_tree[i][j]>target)_tree[i][j]--;
      }
    
      if(_masterParent>target)_masterParent--;
    }
  _pts--;
  //printf("done subtracting %d\n",_pts);
  
}

template <typename T>
void KdTree<T>::_count(int where, int *ct){
//a way to count the number of vital elements on a given branch

  if(_tree[where][1]>=0){
    ct[0]++;
    _count(_tree[where][1],ct);
  }
  if(_tree[where][2]>=0){
    ct[0]++;
    _count(_tree[where][2],ct);
  }
}

template <typename T>
void KdTree<T>::_reassign(int target){
   
   int where,dir,k;
   
   where=_masterParent;
   if(data[target][_tree[where][0]]<data[where][_tree[where][0]])dir=1;
   else dir=2;
   
   k=_tree[where][dir];
   while(k>=0){
     where=k;
     if(data[target][_tree[where][0]]<data[where][_tree[where][0]])dir=1;
     else dir=2;
     k=_tree[where][dir];
   }
   
   _tree[where][dir]=target;
   _tree[target][3]=where;
   _tree[target][1]=-1;
   _tree[target][2]=-1;
   _tree[target][0]=_tree[where][0]+1;
   if(_tree[target][0]==_dimensions)_tree[target][0]=0;
   

}

template <typename T>
void KdTree<T>::_descend(int root){

  if(_tree[root][1]>=0)_descend(_tree[root][1]);
  if(_tree[root][2]>=0)_descend(_tree[root][2]);
  
  _reassign(root);  
    

}

template <typename T>
double KdTree<T>::_distance(ndarray::Array<const T,1,1> const &p1, ndarray::Array<const T,1,1> const &p2)
{

    int i,dd;
    double ans;
    ans=0.0;
    dd=p1.template getSize<0>();
    for(i=0;i<dd;i++)ans+=(p1[i]-p2[i])*(p1[i]-p2[i]);
    return ::sqrt(ans);

}

template <typename T>
GaussianProcess<T>::~GaussianProcess(){
    delete _kdTreePtr;

}

template <typename T>
GaussianProcess<T>::GaussianProcess( 
				    ndarray::Array<T,2,2> const &datain, 
                                    ndarray::Array<T,1,1> const &ff,
                                    boost::shared_ptr< Covariogram<T> > const &covarin)

{
  int i,j;
  ndarray::Array<int,2,2> ndtest;
  
  ndtest=allocate(ndarray::makeVector(3,3));
  
  _covariogram=covarin;
  //_dimensions=dd;
  //_pts=pp;
  
  _pts=datain.template getSize<0>();
  _dimensions=datain.template getSize<1>();
  
  _room=_pts;
  _roomStep=5000;
  
  _nFunctions=1;
  _function=allocate(ndarray::makeVector(_pts,1));
  for(i=0;i<_pts;i++)_function[i][0]=ff[i];
  _krigingParameter=T(1.0);
  
  _lambda=T(1.0e-5);
  
  _useMaxMin=0;
    
  _data=allocate(ndarray::makeVector(_pts,_dimensions));
  
  for(i=0;i<_pts;i++){
     for(j=0;j<_dimensions;j++){
       _data[i][j]=datain[i][j];
     }
  }
  
  _kdTreePtr=new KdTree<T>(_data);
  _data=_kdTreePtr->data;
  _pts=_kdTreePtr->getPoints();
  
}

template <typename T>
GaussianProcess<T>::GaussianProcess( 
				    ndarray::Array<T,2,2> const &datain,
                                    ndarray::Array<T,1,1> const &mn, 
				    ndarray::Array<T,1,1> const &mx, 
				    ndarray::Array<T,1,1> const &ff,
                                    boost::shared_ptr< Covariogram<T> > const &covarin
				    )
{

  int i,j;
  
  _covariogram=covarin;
 // _dimensions=dd;
 // _pts=pp;
 
 _pts=datain.template getSize<0>();
 _dimensions=datain.template getSize<1>();
 
  _room=_pts;
  _roomStep=5000;
  
  _krigingParameter=T(1.0);

 _lambda=T(1.0e-5);
 _krigingParameter=T(1.0);
   
 _max=allocate(ndarray::makeVector(_dimensions));
 _min=allocate(ndarray::makeVector(_dimensions)); 
 _max.deep()=mx;
  _min.deep()=mn;  
 _useMaxMin=1;
 _data=allocate(ndarray::makeVector(_pts,_dimensions));  
  for(i=0;i<_pts;i++){
   
    for(j=0;j<_dimensions;j++){
      _data[i][j]=(datain[i][j]-_min[j])/(_max[j]-_min[j]); //note the normalization by _max-_min in each dimension
    }
  }
   
  _kdTreePtr=new KdTree<T>(_data);
  _data=_kdTreePtr->data;
  _pts=_kdTreePtr->getPoints();  
  _nFunctions=1;
  _function=allocate(ndarray::makeVector(_pts,1));
  for(i=0;i<_pts;i++)_function[i][0]=ff[i];

}

template <typename T>
GaussianProcess<T>::GaussianProcess( 
				    ndarray::Array<T,2,2> const &datain, 
                                    ndarray::Array<T,2,2> const &ff,
                                    boost::shared_ptr< Covariogram<T> > const &covarin)

{
  int i,j;
  ndarray::Array<int,2,2> ndtest;
  
  ndtest=allocate(ndarray::makeVector(3,3));
  
  _covariogram=covarin;
  //_dimensions=dd;
  //_pts=pp;
  
  _pts=datain.template getSize<0>();
  _dimensions=datain.template getSize<1>();
  
  _room=_pts;
  _roomStep=5000;
  
  _nFunctions=ff.template getSize<1>();
  _function=allocate(ndarray::makeVector(_pts,_nFunctions));
  _function.deep()=ff;
  
  _krigingParameter=T(1.0);

  _lambda=T(1.0e-5);
  
  _useMaxMin=0;
    
  _data=allocate(ndarray::makeVector(_pts,_dimensions));
  
  for(i=0;i<_pts;i++){
     for(j=0;j<_dimensions;j++){
       _data[i][j]=datain[i][j];
     }
  }
  
  _kdTreePtr=new KdTree<T>(_data);
  _data=_kdTreePtr->data;
  _pts=_kdTreePtr->getPoints();
  
}

template <typename T>
GaussianProcess<T>::GaussianProcess( 
				    ndarray::Array<T,2,2> const &datain,
                                    ndarray::Array<T,1,1> const &mn, 
				    ndarray::Array<T,1,1> const &mx, 
				    ndarray::Array<T,2,2> const &ff,
                                    boost::shared_ptr< Covariogram<T> > const &covarin
				    )
{

  int i,j;
  
  _covariogram=covarin;
 // _dimensions=dd;
 // _pts=pp;
 
 _pts=datain.template getSize<0>();
 _dimensions=datain.template getSize<1>();
 
  _room=_pts;
  _roomStep=5000;
  
  _krigingParameter=T(1.0);

 _lambda=T(1.0e-5);
 _krigingParameter=T(1.0);
   
 _max=allocate(ndarray::makeVector(_dimensions));
 _min=allocate(ndarray::makeVector(_dimensions)); 
 _max.deep()=mx;
  _min.deep()=mn;  
 _useMaxMin=1;
 _data=allocate(ndarray::makeVector(_pts,_dimensions));  
  for(i=0;i<_pts;i++){
   
    for(j=0;j<_dimensions;j++){
      _data[i][j]=(datain[i][j]-_min[j])/(_max[j]-_min[j]); //note the normalization by _max-_min in each dimension
    }
  }
   
  _kdTreePtr=new KdTree<T>(_data);
  _data=_kdTreePtr->data;
  _pts=_kdTreePtr->getPoints();  
  _nFunctions=ff.template getSize<1>();
  _function=allocate(ndarray::makeVector(_pts,_nFunctions));
  //for(i=0;i<_pts;i++)_function[i][0]=ff[i];
  _function.deep()=ff;

}



template <typename T>
T GaussianProcess<T>::interpolate(ndarray::Array<T,1,1> variance, 
                                  ndarray::Array<T,1,1> const &vin,
                                  int numberOfNeighbors
				  )
{

  int i,j;
  T fbar,mu;
  double before,after,af,bef;
  
  ndarray::Array<T,1,1> covarianceTestPoint;
  ndarray::Array<int,1,1> neighbors;
  ndarray::Array<double,1,1> neighborDistances,vv;
  
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> covariance,bb,xx;
  Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
  
  _timer.interpolationCount++;
  
  before=double(::time(NULL));

   bb.resize(numberOfNeighbors,1);
   xx.resize(numberOfNeighbors,1);
   covariance.resize(numberOfNeighbors,numberOfNeighbors);
   covarianceTestPoint=allocate(ndarray::makeVector(numberOfNeighbors));
   neighbors=allocate(ndarray::makeVector(numberOfNeighbors));;
   neighborDistances=allocate(ndarray::makeVector(numberOfNeighbors));
     
   vv=allocate(ndarray::makeVector(_dimensions));
  
  
  if(_useMaxMin==1){
    //if you constructed this Gaussian process with minimum and maximum values for the dimensions of your parameter space,
    //the point you are interpolating must be scaled to match the data so that the selected nearest neighbors are appropriate
    
    for(i=0;i<_dimensions;i++)vv[i]=(vin[i]-_min[i])/(_max[i]-_min[i]);
  }
  else{
    vv=vin;
  }
  
  bef=double(::time(NULL));
  _kdTreePtr->findNeighbors(neighbors, neighborDistances, vv,
                            numberOfNeighbors);
  af=double(::time(NULL));
  _timer.searchTime+=af-bef;
  
  fbar=0.0;
  for(i=0;i<numberOfNeighbors;i++)fbar+=_function[neighbors[i]][0];
  fbar=fbar/double(numberOfNeighbors);

  for(i=0;i<numberOfNeighbors;i++){
    covarianceTestPoint[i]=(*_covariogram)(vv,_data[neighbors[i]]);
    covariance(i,i)=(*_covariogram)(_data[neighbors[i]],_data[neighbors[i]])\
    +_lambda;
    for(j=i+1;j<numberOfNeighbors;j++){
      covariance(i,j)=(*_covariogram)(_data[neighbors[i]],_data[neighbors[j]]);
      covariance(j,i)=covariance(i,j);
    }
  }
  bef=double(::time(NULL));
  _timer.iterationTime+=bef-af;
  
  //use Eigen's ldlt solver in place of matrix inversion (for speed purposes)
  ldlt.compute(covariance); 
 
  for(i=0;i<numberOfNeighbors;i++)bb(i,0)=_function[neighbors[i]][0]-fbar;
  xx=ldlt.solve(bb);
  af=double(::time(NULL));
  _timer.eigenTime+=af-bef;
  
  mu=fbar;

  for(i=0;i<numberOfNeighbors;i++){
    mu+=covarianceTestPoint[i]*xx(i,0);
  }
  bef=double(::time(NULL));
  _timer.iterationTime+=bef-af;
  
  
  variance(0)=(*_covariogram)(vv,vv)+_lambda;
  
  for(i=0;i<numberOfNeighbors;i++)bb(i)=covarianceTestPoint[i];

  xx=ldlt.solve(bb);
  
  for(i=0;i<numberOfNeighbors;i++){
    variance(0)-=covarianceTestPoint[i]*xx(i,0);
  } 
  
  
  variance(0)=variance(0)*_krigingParameter;
  after=double(::time(NULL));
  _timer.varianceTime+=after-bef;
  _timer.totalTime+=after-before;
  
  return mu;
}

template <typename T>
void GaussianProcess<T>::interpolate(
                                  ndarray::Array<T,1,1> mu,
                                  ndarray::Array<T,1,1> variance, 
                                  ndarray::Array<T,1,1> const &vin,
                                  int numberOfNeighbors
				  )
{

  int i,j,ii;
  T fbar;
  double before,after,af,bef;
  
  ndarray::Array<T,1,1> covarianceTestPoint;
  ndarray::Array<int,1,1> neighbors;
  ndarray::Array<double,1,1> neighborDistances,vv;
  
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> covariance,bb,xx;
  Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
  
  _timer.interpolationCount++;
  before=double(::time(NULL));
  
   bb.resize(numberOfNeighbors,1);
   xx.resize(numberOfNeighbors,1);
   covariance.resize(numberOfNeighbors,numberOfNeighbors);
   covarianceTestPoint=allocate(ndarray::makeVector(numberOfNeighbors));
   neighbors=allocate(ndarray::makeVector(numberOfNeighbors));;
   neighborDistances=allocate(ndarray::makeVector(numberOfNeighbors));
     
   vv=allocate(ndarray::makeVector(_dimensions));
  
  
  if(_useMaxMin==1){
    //if you constructed this Gaussian process with minimum and maximum values for the dimensions of your parameter space,
    //the point you are interpolating must be scaled to match the data so that the selected nearest neighbors are appropriate
    
    for(i=0;i<_dimensions;i++)vv[i]=(vin[i]-_min[i])/(_max[i]-_min[i]);
  }
  else{
    vv=vin;
  }
  
  bef=double(::time(NULL));
  _kdTreePtr->findNeighbors(neighbors, neighborDistances, vv,
                            numberOfNeighbors);
  
  af=double(::time(NULL));
  _timer.searchTime+=af-bef;
  
  for(i=0;i<numberOfNeighbors;i++){
    covarianceTestPoint[i]=(*_covariogram)(vv,_data[neighbors[i]]);
    covariance(i,i)=(*_covariogram)(_data[neighbors[i]],_data[neighbors[i]])\
    +_lambda;
    for(j=i+1;j<numberOfNeighbors;j++){
      covariance(i,j)=(*_covariogram)(_data[neighbors[i]],_data[neighbors[j]]);
      covariance(j,i)=covariance(i,j);
    }
  }
  bef=double(::time(NULL));
  _timer.iterationTime+=bef-af;
  
  //use Eigen's ldlt solver in place of matrix inversion (for speed purposes)
  ldlt.compute(covariance); 
  af=double(::time(NULL));
  _timer.eigenTime+=af-bef;
  
    for(ii=0;ii<_nFunctions;ii++){
  
      fbar=0.0;
      for(i=0;i<numberOfNeighbors;i++)fbar+=_function[neighbors[i]][ii];
      fbar=fbar/double(numberOfNeighbors);
      
      for(i=0;i<numberOfNeighbors;i++)bb(i,0)=_function[neighbors[i]][ii]-fbar;
      xx=ldlt.solve(bb);
      
      mu[ii]=fbar;

      for(i=0;i<numberOfNeighbors;i++){
        mu[ii]+=covarianceTestPoint[i]*xx(i,0);
      }
  
     
  }//ii=0 through _nFunctions
  bef=double(::time(NULL));
  _timer.eigenTime+=bef-af;
  
   variance[0]=(*_covariogram)(vv,vv)+_lambda;
  
   for(i=0;i<numberOfNeighbors;i++)bb(i)=covarianceTestPoint[i];

   xx=ldlt.solve(bb);
  
  

   for(i=0;i<numberOfNeighbors;i++){
     variance[0]-=covarianceTestPoint[i]*xx(i,0);
   } 
   
  
  
   variance[0]=variance[0]*_krigingParameter;
  
  
  for(i=1;i<_nFunctions;i++)variance[i]=variance[0];
  after=double(::time(NULL));
  _timer.varianceTime+=after-bef;
  _timer.totalTime+=after-before;
}


template <typename T>
T GaussianProcess<T>::selfInterpolate(ndarray::Array<T,1,1> variance, int dex, int numberOfNeighbors){
  
  int i,j;
  T fbar,mu;
  double before,after,af,bef;

  ndarray::Array<T,1,1> covarianceTestPoint;
  ndarray::Array<int,1,1> selfNeighbors;
  ndarray::Array<double,1,1> selfDistances;
  ndarray::Array<int,1,1> neighbors;
  ndarray::Array<double,1,1> neighborDistances;
  
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> covariance,bb,xx;
  Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
  
  _timer.interpolationCount++;
  
  before=double(::time(NULL));
  
  bb.resize(numberOfNeighbors,1);
  xx.resize(numberOfNeighbors,1);
  covariance.resize(numberOfNeighbors,numberOfNeighbors);
  covarianceTestPoint=allocate(ndarray::makeVector(numberOfNeighbors));
  neighbors=allocate(ndarray::makeVector(numberOfNeighbors));
  neighborDistances=allocate(ndarray::makeVector(numberOfNeighbors));
     
  selfNeighbors=allocate(ndarray::makeVector(numberOfNeighbors+1));
  selfDistances=allocate(ndarray::makeVector(numberOfNeighbors+1));
  
  //we don't use _useMaxMin because _data has already been normalized
  
  bef=double(::time(NULL));
  _kdTreePtr->findNeighbors(selfNeighbors, selfDistances, _data[dex], 
                            numberOfNeighbors+1);
  af=double(::time(NULL));
  _timer.searchTime=af-bef;
  
  if(selfNeighbors[0]!=dex){
    std::cout<<"WARNING selfdist "<<selfDistances[0]<<" "<<selfDistances[1]<<"\n";
    std::cout<<"dex "<<dex<<" "<<selfNeighbors[0]<<"\n";
    exit(1);
  }
  
  //SelfNeighbors[0] will be the point itself (it is its own nearest neighbor)
  //We discard that for the interpolation calculation
  //
  //If you do not wish to do this, simply call the usual ::interpolate() method instead of
  //::selfInterpolate()
  for(i=0;i<numberOfNeighbors;i++){
    neighbors[i]=selfNeighbors[i+1];
    neighborDistances[i]=selfDistances[i+1];
  }

  fbar=0.0;
  for(i=0;i<numberOfNeighbors;i++)fbar+=_function[neighbors[i]][0];
  fbar=fbar/double(numberOfNeighbors);

  for(i=0;i<numberOfNeighbors;i++){
    covarianceTestPoint[i]=(*_covariogram)(_data[dex],_data[neighbors[i]]);
    covariance(i,i)=(*_covariogram)(_data[neighbors[i]],_data[neighbors[i]])
    +_lambda;
    for(j=i+1;j<numberOfNeighbors;j++){
      covariance(i,j)=(*_covariogram)(_data[neighbors[i]],_data[neighbors[j]]);
      covariance(j,i)=covariance(i,j);
    }
  }
  bef=double(::time(NULL));
  _timer.iterationTime+=bef-af;

  
  //use Eigen's ldlt solver in place of matrix inversion (for speed purposes)
  ldlt.compute(covariance); 
  
  
  for(i=0;i<numberOfNeighbors;i++)bb(i,0)=_function[neighbors[i]][0]-fbar;
  xx=ldlt.solve(bb);
  af=double(::time(NULL));
  _timer.eigenTime+=af-bef;
  
  mu=fbar;

  for(i=0;i<numberOfNeighbors;i++){
    mu+=covarianceTestPoint[i]*xx(i,0);
  }
  
  bef=double(::time(NULL));
  variance(0)=(*_covariogram)(_data[dex],_data[dex])+_lambda;
  
  for(i=0;i<numberOfNeighbors;i++)bb(i)=covarianceTestPoint[i];
  
  
 
  xx=ldlt.solve(bb);
  
  for(i=0;i<numberOfNeighbors;i++){
    variance(0)-=covarianceTestPoint[i]*xx(i,0);
  } 
 
  
  variance(0)=variance(0)*_krigingParameter;
  after=double(::time(NULL));
  _timer.varianceTime+=after-bef;
  _timer.totalTime+=after-before;
  
  return mu;
}

template <typename T>
void GaussianProcess<T>::selfInterpolate(
                                         ndarray::Array<T,1,1> mu, 
                                         ndarray::Array<T,1,1> variance, int dex, int numberOfNeighbors){
  
  int i,j,ii;
  T fbar;
  double before,after,af,bef;

  ndarray::Array<T,1,1> covarianceTestPoint;
  ndarray::Array<int,1,1> selfNeighbors;
  ndarray::Array<double,1,1> selfDistances;
  ndarray::Array<int,1,1> neighbors;
  ndarray::Array<double,1,1> neighborDistances;
  
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> covariance,bb,xx;
  Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
  
  _timer.interpolationCount++;
  before=double(::time(NULL));

  bb.resize(numberOfNeighbors,1);
  xx.resize(numberOfNeighbors,1);
  covariance.resize(numberOfNeighbors,numberOfNeighbors);
  covarianceTestPoint=allocate(ndarray::makeVector(numberOfNeighbors));
  neighbors=allocate(ndarray::makeVector(numberOfNeighbors));
  neighborDistances=allocate(ndarray::makeVector(numberOfNeighbors));
     
  selfNeighbors=allocate(ndarray::makeVector(numberOfNeighbors+1));
  selfDistances=allocate(ndarray::makeVector(numberOfNeighbors+1));
  
  //we don't use _useMaxMin because _data has already been normalized
  
  bef=double(::time(NULL));
   _kdTreePtr->findNeighbors(selfNeighbors, selfDistances, _data[dex], 
                            numberOfNeighbors+1);
  
  af=double(::time(NULL));
  _timer.searchTime+=af-bef;
  
  if(selfNeighbors[0]!=dex){
    std::cout<<"WARNING selfdist "<<selfDistances[0]<<" "<<selfDistances[1]<<"\n";
    std::cout<<"dex "<<dex<<" "<<selfNeighbors[0]<<"\n";
    exit(1);
  }
  
  //SelfNeighbors[0] will be the point itself (it is its own nearest neighbor)
  //We discard that for the interpolation calculation
  //
  //If you do not wish to do this, simply call the usual ::interpolate() method instead of
  //::selfInterpolate()
  for(i=0;i<numberOfNeighbors;i++){
    neighbors[i]=selfNeighbors[i+1];
    neighborDistances[i]=selfDistances[i+1];
  }



  for(i=0;i<numberOfNeighbors;i++){
    covarianceTestPoint[i]=(*_covariogram)(_data[dex],_data[neighbors[i]]);
    covariance(i,i)=(*_covariogram)(_data[neighbors[i]],_data[neighbors[i]])
    +_lambda;
    for(j=i+1;j<numberOfNeighbors;j++){
      covariance(i,j)=(*_covariogram)(_data[neighbors[i]],_data[neighbors[j]]);
      covariance(j,i)=covariance(i,j);
    }
  }
  bef=double(::time(NULL));
  _timer.iterationTime+=bef-af;  

  
  //use Eigen's ldlt solver in place of matrix inversion (for speed purposes)
  ldlt.compute(covariance); 
  
  for(ii=0;ii<_nFunctions;ii++){
  
    fbar=0.0;
    for(i=0;i<numberOfNeighbors;i++)fbar+=_function[neighbors[i]][ii];
    fbar=fbar/double(numberOfNeighbors);
  
    for(i=0;i<numberOfNeighbors;i++)bb(i,0)=_function[neighbors[i]][ii]-fbar;
    xx=ldlt.solve(bb);
 
    mu[ii]=fbar;

    for(i=0;i<numberOfNeighbors;i++){
      mu[ii]+=covarianceTestPoint[i]*xx(i,0);
    }
  }//ii=0 through _nFunctions
  
  af=double(::time(NULL));
  _timer.eigenTime+=af-bef;
  
  variance[0]=(*_covariogram)(_data[dex],_data[dex])+_lambda;
  
  for(i=0;i<numberOfNeighbors;i++)bb(i)=covarianceTestPoint[i];

  xx=ldlt.solve(bb);

  
  for(i=0;i<numberOfNeighbors;i++){
    variance[0]-=covarianceTestPoint[i]*xx(i,0);
  } 
  
  variance[0]=variance[0]*_krigingParameter;
  
  
  for(i=1;i<_nFunctions;i++)variance[i]=variance[0];
  
  after=double(::time(NULL));
  _timer.varianceTime+=after-af;
  _timer.totalTime+=after-before;
}


template<typename T>
void GaussianProcess<T>::batchInterpolate(ndarray::Array<T,1,1> mu, \
ndarray:: Array<T,1,1> variance, ndarray::Array<T,2,2> const &queries){
    
  int i,j,ii,nQueries;
  double af,bef,before,after;
  T fbar;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> batchCovariance,batchbb,batchxx;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> queryCovariance;
  Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
  
  ndarray::Array<T,1,1> v1; 

  before=double(::time(NULL));
  
  nQueries=queries.template getSize<0>();
  
  _timer.interpolationCount+=nQueries;
  
  v1=allocate(ndarray::makeVector(_dimensions));
  batchbb.resize(_pts,1);
  batchxx.resize(_pts,1);
  batchCovariance.resize(_pts,_pts);
  queryCovariance.resize(_pts,1);
 
  
  for(i=0;i<_pts;i++){
    
    batchCovariance(i,i)=(*_covariogram)(_data[i],_data[i])+_lambda;
    for(j=i+1;j<_pts;j++){
      batchCovariance(i,j)=(*_covariogram)(_data[i],_data[j]);
      batchCovariance(j,i)=batchCovariance(i,j);
    }
  }
  af=double(::time(NULL));
  _timer.iterationTime+=af-before;
  
  ldlt.compute(batchCovariance);  
  
  
  fbar=0.0;
  for(i=0;i<_pts;i++){
    fbar+=_function[i][0];
  }
  fbar=fbar/T(_pts);
  
  //std::cout<<"fbar "<<fbar<<"\n";
  
  for(i=0;i<_pts;i++){
    batchbb(i,0)=_function[i][0]-fbar;
  }
  batchxx=ldlt.solve(batchbb);
  bef=double(::time(NULL));
  _timer.eigenTime+=bef-af;
  
  
  for(ii=0;ii<nQueries;ii++){
    for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
    if(_useMaxMin==1){
      for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
    } 
    mu(ii)=fbar;
    for(i=0;i<_pts;i++){
      mu(ii)+=batchxx(i)*(*_covariogram)(v1,_data[i]);
    }
  }
  af=double(::time(NULL));
  _timer.iterationTime+=af-bef;
  
  //std::cout<<"done with interpolation\n";
  
  for(ii=0;ii<nQueries;ii++){
    //std::cout<<"i "<<ii<<"\n";
    for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
    if(_useMaxMin==1){
      for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
    }
    
    for(i=0;i<_pts;i++){
      batchbb(i,0)=(*_covariogram)(v1,_data[i]);
      queryCovariance(i,0)=batchbb(i,0);
    }
    batchxx=ldlt.solve(batchbb);
    
    variance(ii)=(*_covariogram)(v1,v1)+_lambda;
    
    for(i=0;i<_pts;i++){
      variance(ii)-=queryCovariance(i,0)*batchxx(i);
    }
    
    variance(ii)=variance(ii)*_krigingParameter;
      
  }
  after=double(::time(NULL));
  _timer.varianceTime+=after-af;
  _timer.totalTime+=after-before;


}

template<typename T>
void GaussianProcess<T>::batchInterpolate(ndarray::Array<T,2,2> mu, \
ndarray:: Array<T,2,2> variance, ndarray::Array<T,2,2> const &queries){
    
  int i,j,ii,nQueries,ifn;
  double af,bef,before,after;
  T fbar;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> batchCovariance,batchbb,batchxx;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> queryCovariance;
  Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
  
  ndarray::Array<T,1,1> v1; 

  before=double(::time(NULL));
  
  nQueries=queries.template getSize<0>();
  
  _timer.interpolationCount+=nQueries;
  
  v1=allocate(ndarray::makeVector(_dimensions));
  batchbb.resize(_pts,1);
  batchxx.resize(_pts,1);
  batchCovariance.resize(_pts,_pts);
  queryCovariance.resize(_pts,1);
  
  for(i=0;i<_pts;i++){
    
    batchCovariance(i,i)=(*_covariogram)(_data[i],_data[i])+_lambda;
    for(j=i+1;j<_pts;j++){
      batchCovariance(i,j)=(*_covariogram)(_data[i],_data[j]);
      batchCovariance(j,i)=batchCovariance(i,j);
    }
  }
  
  af=double(::time(NULL));
  _timer.iterationTime+=af-before;
  
  ldlt.compute(batchCovariance);  
  
  bef=double(::time(NULL));
  _timer.eigenTime+=bef-af;
  
  for(ifn=0;ifn<_nFunctions;ifn++){
      fbar=0.0;
      for(i=0;i<_pts;i++){
        fbar+=_function[i][ifn];
      }
      fbar=fbar/T(_pts);
  
      //std::cout<<"fbar "<<fbar<<"\n";
      
      bef=double(::time(NULL));
      for(i=0;i<_pts;i++){
        batchbb(i,0)=_function[i][ifn]-fbar;
      }
      batchxx=ldlt.solve(batchbb);
      af=double(::time(NULL));
      _timer.eigenTime+=af-bef;
      
      bef=double(::time(NULL));
      for(ii=0;ii<nQueries;ii++){
        for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
        if(_useMaxMin==1){
          for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
        } 
        mu[ii][ifn]=fbar;
        for(i=0;i<_pts;i++){
          mu[ii][ifn]+=batchxx(i)*(*_covariogram)(v1,_data[i]);
        }
      }
      af=double(::time(NULL));
      _timer.iterationTime+=af-bef;

  }//ifn=0 to _nFunctions
  
  
  //std::cout<<"done with interpolation\n";
  
  bef=double(::time(NULL));
  for(ii=0;ii<nQueries;ii++){
    //std::cout<<"i "<<ii<<"\n";
    for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
    if(_useMaxMin==1){
      for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
    }
    
    for(i=0;i<_pts;i++){
      batchbb(i,0)=(*_covariogram)(v1,_data[i]);
      queryCovariance(i,0)=batchbb(i,0);
    }
    batchxx=ldlt.solve(batchbb);
    
    variance[ii][0]=(*_covariogram)(v1,v1)+_lambda;
    
    for(i=0;i<_pts;i++){
      variance[ii][0]-=queryCovariance(i,0)*batchxx(i);
    }
    
    variance[ii][0]=variance[ii][0]*_krigingParameter;
    for(i=1;i<_nFunctions;i++)variance[ii][i]=variance[ii][0];
      
  }
  after=double(::time(NULL));
  _timer.varianceTime+=after-bef;
  _timer.totalTime+=after-before;

}

template<typename T>
void GaussianProcess<T>::batchInterpolate(ndarray::Array<T,1,1> mu, ndarray::Array<T,2,2> const &queries){

  int i,j,ii,nQueries;
  double af,bef,before,after;

  T fbar;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> batchCovariance,batchbb,batchxx;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> queryCovariance;
  Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
  
  ndarray::Array<T,1,1> v1;

  before=double(::time(NULL));
  
  nQueries=queries.template getSize<0>();
  
  _timer.interpolationCount+=nQueries;
  
  v1=allocate(ndarray::makeVector(_dimensions));
  
  batchbb.resize(_pts,1);
  batchxx.resize(_pts,1);
  batchCovariance.resize(_pts,_pts);
  queryCovariance.resize(_pts,1);
 
  
  for(i=0;i<_pts;i++){
    batchCovariance(i,i)=(*_covariogram)(_data[i],_data[i])+_lambda;
    for(j=i+1;j<_pts;j++){
      batchCovariance(i,j)=(*_covariogram)(_data[i],_data[j]);
      batchCovariance(j,i)=batchCovariance(i,j);
    }
  }
  af=double(::time(NULL));
  _timer.iterationTime+=af-before;
  
  ldlt.compute(batchCovariance);  

  fbar=0.0;
  for(i=0;i<_pts;i++){
    fbar+=_function[i][0];
  }
  fbar=fbar/T(_pts);
  
  //std::cout<<"fbar "<<fbar<<"\n";
  
  for(i=0;i<_pts;i++){
    batchbb(i,0)=_function[i][0]-fbar;
  }
  batchxx=ldlt.solve(batchbb);
  bef=double(::time(NULL));
  _timer.eigenTime+=bef-af;

  for(ii=0;ii<nQueries;ii++){
    for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
    if(_useMaxMin==1){
      for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
    }
    
    mu(ii)=fbar;
    for(i=0;i<_pts;i++){
      mu(ii)+=batchxx(i)*(*_covariogram)(v1,_data[i]);
    }
  }
  after=double(::time(NULL));
  _timer.iterationTime+=after-bef;
  _timer.totalTime+=after-before;

  
  //std::cout<<"done with interpolation\n";
}

template<typename T>
void GaussianProcess<T>::batchInterpolate(ndarray::Array<T,2,2> mu, ndarray::Array<T,2,2> const &queries){

  int i,j,ii,nQueries,ifn;
  double af,bef,before,after;

  T fbar;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> batchCovariance,batchbb,batchxx;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> queryCovariance;
  Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
  
  ndarray::Array<T,1,1> v1;
 
  before=double(::time(NULL));
 
  nQueries=queries.template getSize<0>();
  
  _timer.interpolationCount+=nQueries;
  
  v1=allocate(ndarray::makeVector(_dimensions));
  
  batchbb.resize(_pts,1);
  batchxx.resize(_pts,1);
  batchCovariance.resize(_pts,_pts);
  queryCovariance.resize(_pts,1);
 
  
  for(i=0;i<_pts;i++){
    batchCovariance(i,i)=(*_covariogram)(_data[i],_data[i])+_lambda;
    for(j=i+1;j<_pts;j++){
      batchCovariance(i,j)=(*_covariogram)(_data[i],_data[j]);
      batchCovariance(j,i)=batchCovariance(i,j);
    }
  }
  
  af=double(::time(NULL));
  _timer.iterationTime+=af-before;
  
  ldlt.compute(batchCovariance);  
  
  bef=double(::time(NULL));
  _timer.eigenTime+=bef-af;
  
  for(ifn=0;ifn<_nFunctions;ifn++){
      fbar=0.0;
      for(i=0;i<_pts;i++){
        fbar+=_function[i][ifn];
      }
      fbar=fbar/T(_pts);
  
      //std::cout<<"fbar "<<fbar<<"\n";
      bef=double(::time(NULL));
      for(i=0;i<_pts;i++){
        batchbb(i,0)=_function[i][ifn]-fbar;
      }
      batchxx=ldlt.solve(batchbb);
      af=double(::time(NULL));
      _timer.eigenTime+=af-bef;
  
     
      for(ii=0;ii<nQueries;ii++){
        for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
        if(_useMaxMin==1){
          for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
        }
    
        mu[ii][ifn]=fbar;
        for(i=0;i<_pts;i++){
          mu[ii][ifn]+=batchxx(i)*(*_covariogram)(v1,_data[i]);
        }
      }
      bef=double(::time(NULL));
      _timer.iterationTime+=bef-af;
      
  }//ifn=0 through _nFunctions
  
  after=double(::time(NULL));
  _timer.totalTime+=after-before;
  
  //std::cout<<"done with interpolation\n";
}


template <typename T>
void GaussianProcess<T>::addPoint(ndarray::Array<T,1,1> const &vin, T f){

  int i,j;
  
  if(_nFunctions!=1){
    std::cout<<"_nFunctions is "<<_nFunctions<<"\n";
    std::cout<<"you should call addPoint passing an ndarray for f\n";
    return;
  }
  
  ndarray::Array<T,1,1> v;
  v=allocate(ndarray::makeVector(_dimensions));
 
  for(i=0;i<_dimensions;i++){
    v[i]=vin[i];
    if(_useMaxMin==1){
      v[i]=(v[i]-_min[i])/(_max[i]-_min[i]);
    }
    
  }
  
  if(_pts==_room){
    ndarray::Array<T,2,2> buff;
    buff=allocate(ndarray::makeVector(_pts,_nFunctions));
    buff.deep()=_function;
    
    _room+=_roomStep;
    _function=allocate(ndarray::makeVector(_room,_nFunctions));
    for(i=0;i<_pts;i++){
      for(j=0;j<_nFunctions;j++){
         _function[i][j]=buff[i][j];
      }
    }
 
  }
  _function[_pts][0]=f;
  
  _kdTreePtr->addPoint(v);
  _pts=_kdTreePtr->getPoints();
  _data=_kdTreePtr->data;
  

}

template <typename T>
void GaussianProcess<T>::addPoint(ndarray::Array<T,1,1> const &vin, 
                                  ndarray::Array<T,1,1> const &f){

  int i,j;
  
  ndarray::Array<T,1,1> v;
  v=allocate(ndarray::makeVector(_dimensions));
 
  for(i=0;i<_dimensions;i++){
    v[i]=vin[i];
    if(_useMaxMin==1){
      v[i]=(v[i]-_min[i])/(_max[i]-_min[i]);
    }
    
  }
  
  if(_pts==_room){
    ndarray::Array<T,2,2> buff;
    buff=allocate(ndarray::makeVector(_pts,_nFunctions));
    buff.deep()=_function;
    
    _room+=_roomStep;
    _function=allocate(ndarray::makeVector(_room,_nFunctions));
    for(i=0;i<_pts;i++){
      for(j=0;j<_nFunctions;j++){
         _function[i][j]=buff[i][j];
      }
    }
 
  }
  for(i=0;i<_nFunctions;i++)_function[_pts][i]=f[i];
  
  _kdTreePtr->addPoint(v);
  _pts=_kdTreePtr->getPoints();
  _data=_kdTreePtr->data;
  

}

template <typename T>
void GaussianProcess<T>::removePoint(int dex){
  
  int i,j;
  
  _kdTreePtr->remove(dex);
  
  for(i=dex;i<_pts;i++){
      for(j=0;j<_nFunctions;j++){
          _function[i][j]=_function[i+1][j];  
      }
  }
  _pts=_kdTreePtr->getPoints();
}

template <typename T>
void GaussianProcess<T>::setKrigingParameter(T kk){
  
  _krigingParameter=kk;
  
}

template <typename T>
void GaussianProcess<T>::setCovariogram(boost::shared_ptr< Covariogram<T> > const &covar){
   _covariogram=covar;
}

template <typename T>
void GaussianProcess<T>::setLambda(T lambda){

  _lambda=lambda;

}


template <typename T>
GaussianProcessTimer& GaussianProcess<T>::getTimes(){
  return _timer;
}



template <typename T>
Covariogram<T>::~Covariogram(){};

/*template <typename T>
Covariogram<T>::Covariogram()
{
   lsst::daf::base::Citizen(typeid(this));
  _nHyperParameters=0;
}

template <typename T>
Covariogram<T>::Covariogram(ndarray::Array<T,1,1> const &input)
{
   lsst::daf::base::Citizen(typeid(this));
  _nHyperParameters=0;
}*/

template <typename T>
void Covariogram<T>::setHyperParameters(ndarray::Array<T,1,1> const &input)
{
    int i;
    
    for(i=0;i<_nHyperParameters;i++)_hyperParameters[i]=input[i];
}

template <typename T>
T Covariogram<T>::operator()(ndarray::Array<const T,1,1> const &p1,
                             ndarray::Array<const T,1,1> const &p2) const
{
    std::cout<<"by the way, you are calling the wrong operator\n";
    exit(1);
    return T(1.0);
}

template <typename T>
void Covariogram<T>::explainHyperParameters(){
    std::cout<<"\nThis is a base class Covariogram; it won't work\n";
}

template <typename T>
SquaredExpCovariogram<T>::~SquaredExpCovariogram(){}

template <typename T>
SquaredExpCovariogram<T>::SquaredExpCovariogram()
{
    Covariogram<T>::_nHyperParameters=1;
    Covariogram<T>::_hyperParameters=allocate(
                  ndarray::makeVector(Covariogram<T>::_nHyperParameters));
    Covariogram<T>::_hyperParameters[0]=1.0;
}

template <typename T>
SquaredExpCovariogram<T>::SquaredExpCovariogram(
                       ndarray::Array<T,1,1> const &initParams)
{
    Covariogram<T>::_nHyperParameters=1;
    Covariogram<T>::_hyperParameters=
               allocate(ndarray::makeVector(Covariogram<T>::_nHyperParameters));
    Covariogram<T>::_hyperParameters[0]=initParams[0];
}

template <typename T>
T SquaredExpCovariogram<T>::operator()(
                            ndarray::Array<const T,1,1> const &p1,
                            ndarray::Array<const T,1,1> const &p2
                            ) const
{
    int i;
    T d;
    d=0.0;
    for(i=0;i<p1.template getSize<0>();i++){
        d+=(p1[i]-p2[i])*(p1[i]-p2[i]);
    }
    d=d/Covariogram<T>::_hyperParameters[0];
    return T(exp(-0.5*d));
}

template <typename T>
void SquaredExpCovariogram<T>::explainHyperParameters(){
    std::cout<<"\nThis is the squared exponential covariogram\n";
    std::cout<<"_nHyperParameters is "<<Covariogram<T>::_nHyperParameters<<"\n";
    std::cout<<"The 0th hyper parameter is the squared length scale ell^2\n";
    std::cout<<"as in C(p1,p2) = exp[-0.5 * (|p1-p2|^2 / ell^2\n";
    std::cout<<"Current value is \n"<<Covariogram<T>::_hyperParameters[0];
}

template <typename T>
NeuralNetCovariogram<T>::~NeuralNetCovariogram(){}

template <typename T>
NeuralNetCovariogram<T>::NeuralNetCovariogram(){
    
    Covariogram<T>::_nHyperParameters=2;
    Covariogram<T>::_hyperParameters=allocate(ndarray::makeVector(2));
    Covariogram<T>::_hyperParameters[0]=1.0;
    Covariogram<T>::_hyperParameters[1]=1.0;
}

template <typename T>
NeuralNetCovariogram<T>::NeuralNetCovariogram(ndarray::Array<T,1,1> const &initparams)
{
    int i;
    Covariogram<T>::_nHyperParameters=2;
    Covariogram<T>::_hyperParameters=allocate(ndarray::makeVector(2));
    for(i=0;i<2;i++){
      Covariogram<T>::_hyperParameters[i]=initparams[i];
    }
}

template <typename T>
T NeuralNetCovariogram<T>::operator()(ndarray::Array<const T,1,1> const &p1,
                                      ndarray::Array<const T,1,1> const &p2
                                      ) const
{
    int i,dim;
    double num,denom1,denom2,arg;    

    dim=p1.template getSize<0>();
    
    num=2.0*Covariogram<T>::_hyperParameters[0];
    denom1=1.0+2.0*Covariogram<T>::_hyperParameters[0];
    denom2=1.0+2.0*Covariogram<T>::_hyperParameters[0];
    for(i=0;i<dim;i++){
        num+=2.0*p1[i]*p2[i]*Covariogram<T>::_hyperParameters[1];
        denom1+=2.0*p1[i]*p1[i]*Covariogram<T>::_hyperParameters[1];
        denom2+=2.0*p2[i]*p2[i]*Covariogram<T>::_hyperParameters[1];
    }
    arg=num/::sqrt(denom1*denom2);
    return T(2.0*(::asin(arg))/3.141592654);

}

template <typename T>
void NeuralNetCovariogram<T>::explainHyperParameters()
{
    std::cout<<"\nThis is the covariogram of a neural network with infinite hidden layers\n";
    std::cout<<"See Rasmussen and Williams (2006) http://www.gaussianprocess.org/gpml/  equation 4.29\n";
    std::cout<<"There are 2 _hyperParameters\n";
    std::cout<<"The 0th hyper parameter is sigma_0 from the reference above; current value - "<<
               Covariogram<T>::_hyperParameters[0]<<"\n";
    std::cout<<"The 1st hyper paramter is sigma from the reference above; current value - "<<
               Covariogram<T>::_hyperParameters[1]<<"\n";
}

}}}
#define gpn lsst::afw::math

#define INSTANTIATEGP(T) \
        template class gpn::KdTree<T>; \
        template class gpn::GaussianProcess<T>; \
        template class gpn::Covariogram<T>; \
        template class gpn::SquaredExpCovariogram<T>;\
        template class gpn::NeuralNetCovariogram<T>; 

INSTANTIATEGP(double);


