# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 23:25:54 2023

@author: LEN
"""
#this is a c++ file code
#include<iostream>
using namespace std;
/*int max(int a,int b,int c)
{
    if(a>b && a>c)
    {return a;}
    else if(b>a && b>c)
    {return b;}
    else 
     {return c;}*/

int main()
{
    int m,n,i,j;
    string a,b;
    cout<<"1st string size:";
    cin>>a;
    cout<<"2nd string size:";
    cin>>b;
    m=a.length();
    n=b.length();
    int x[m+1][n+1];
    for(i=0;i<=m;i++)
    {
        for(j=0;j<=n;j++)
        {
            x[i][j]=0;
            
        }
    }
   int e,d,f;
    for(i=1;i<=m;i++)
    {
        for(j=1;j<=n;j++)
        {
            cout<<i<<j<<x[i][j]<<endl;
            if(a[i-1]== b[j-1])
            {
                x[i][j]=x[i-1][j-1]+2;
            }
            else
            {
               e=x[i-1][j-1],d=x[i][j-1],f=x[i-1][j];
               e--;
               f--;
               d--;
               int c=max(e,f);
               c=max(c,d);
               if(c<0)
               {x[i][j]=0;}
               else
               //{x[i+1][j+1]=c;}
               {x[i][j]=c;}
             }
             //cout<<i<<j<<x[i][j]<<endl;
        }
    }
    cout<<"Score Matrix:"<<"\n";
    for(i=0;i<=m;i++)
    {
        for(j=0;j<=n;j++)
         cout<<x[i][j]<<" ";
        cout<<"\n";
    }
    return 0;
        
}