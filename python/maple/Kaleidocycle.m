(* ::Package:: *)

(* ::Input::Initialization:: *)
(* setup *)
Setup[N_:7,ori_:False,sl_:0, freeangle_:False]:=Module[{},
Clear[h];oriented=ori;slide = sl;n=N;FreeAngle=freeangle;
H=Array[h,{n+1,3}]; (* hinge vectors *)
vars=Join[Table[h[i,1],{i,3,n}],Table[h[i,2],{i,2,n}], Table[h[i,3],{i,2,n-1}]];
(* initial and final hinges are fixed to kill the global symmetry *)
h[1,1]=0;h[1,2]=0;h[1,3]=1;h[2,1]=0;
If[oriented, h[n+1,1]=0;h[n+1,2]=0; h[n+1,3]=1;, 
h[n+1,1]=0;h[n+1,2]=0; h[n+1,3]=-1];
If[FreeAngle,
vars=Join[vars,{h[n,3]}],
(* constraint for constant torsion *)InProd=Table[H[[i]] . H[[i+1]]==h[2,3], {i,1,n}]; 
If[oriented,  h[n,3]=h[2,3],h[n,3]=-h[2,3] ]
];
(* constraint for unit norm *)
Nrm=Table[H[[i]] . H[[i]]==1, {i,1,n}];
(* constraint for closedness *)
If[FreeAngle,
MA=Table[Cross[H[[i]],H[[i+1]]]/Norm[Cross[H[[i]],H[[i+1]]]] ,{i,1,n}],
MA=Table[Cross[H[[i]],H[[i+1]]],{i,1,n}]
];
M:=Table[MA[[i]]+ slide*H[[i]],{i,1,n}];
ExtP=Total[M];
(* combined all constraints *)
If[FreeAngle,
cons = Join[Nrm, {ExtP[[1]]==0, ExtP[[2]]==0, ExtP[[3]]==0}],
cons = Join[InProd, Nrm, {ExtP[[1]]==0, ExtP[[2]]==0, ExtP[[3]]==0}];
];
(* definition for various energies *) 
        (* TODO: when FreeAngle, we do not have to divide by Norm *)
energyE2=Total[Table[Abs[ArcCos[ (M[[i]] . M[[Mod[i+1,n,1]]])/(Norm[M[[i]]]*Norm[M[[Mod[i+1,n,1]]]])]]^2,{i,1,n}]];
(* Bobenko-Suris's bending energy *)
energyE=Total[Table[Log[2]-Log[1+(M[[i]] . M[[Mod[i+1,n,1]]])/(Norm[M[[i]]]*Norm[M[[Mod[i+1,n,1]]]])],{i,1,n}]];
(* spring energy *)
energyS=Total[Table[Log[2]-Log[1+(MA[[i]] . MA[[Mod[i+1,n,1]]])/(Norm[MA[[i]]]*Norm[MA[[Mod[i+1,n,1]]]])],{i,1,n}]];energyB = Total[Table[Cross[M[[i]],M[[Mod[i+1,n,1]]]/(Norm[M[[i]]]*Norm[M[[Mod[i+1,n,1]]]]+M[[i]] . M[[Mod[i+1,n,1]]])],{i,1,n}]];
(* torsion/twist energy *)
energyT2 = Total[Table[Abs[ArcCos[H[[i]] . H[[i+1]]]]^2,{i,1,n}]];
energyT = Total[Table[Log[2]-Log[1+(H[[i]] . H[[i+1]])/(Norm[H[[i]]]*Norm[H[[i+1]]])],{i,1,n}]];
HA=Total[Table[H[[i]], {i,1,n}]];
       Print["Setup ",n," tetrahedra system with slide=",slide," oriented=",oriented, " free (non-constant) twist angle=", FreeAngle];
];
(* random hinge configuration used as initial points *)
RndH:=Module[{r,ip},
ip={};
For[i=3,i<=n,i++,
r=Normalize@RandomVariate[NormalDistribution[],3];
ip = Join[ip,{{h[i,1],r[[1]]},{h[i,2],r[[2]]},{h[i,3],r[[3]]}}];
];
If[Not[FreeAngle],ip=Join[Drop[ip,-1]]];ip=Join[ip,{{h[2,2],Sqrt[1-r[[3]]^2]},{h[2,3],r[[3]]}}];
Return[ip];
];
(* find Kaleidocycle that minimises a functional (with a specified cos if a\[GreaterEqual]-1 is given) *)
Options[MKn]={"boxConstraint"->False,"initialPoint"->0,"torsion"->-2,"maxiter"->500};
MKn[target_:h[2,3],localop_:False, OptionsPattern[]]:=Module[{sols,box,ip},
box = cons;
If[OptionValue["boxConstraint"],
box=Join[box,Table[-1<M[[i]] . M[[Mod[i+1,n,1]]]/(Norm[M[[i]]]*Norm[M[[Mod[i+1,n,1]]]])<1, {i,1,n}]]];
If[OptionValue["torsion"]>=-1, box=Join[box,{h[2,3]==OptionValue["torsion"]}]];
ip=RndH;
If[Head[OptionValue["initialPoint"]]==List,ip=OptionValue["initialPoint"]];
If[localop,
(* for local minimum *)
Print["Finding local minima..."];
sols = FindMinimum[{target,cons},ip,MaxIterations->OptionValue["maxiter"]],
(* for global minimum, use the below (slow) *)
Print["Finding global minima..."];
sols = NMinimize[{target,cons},vars, Method->{"RandomSearch","PostProcess" ->"InteriorPoint"}, WorkingPrecision->14, MaxIterations->OptionValue["maxiter"]]
];
If[Head[sols]==List,PrintSol[sols],Print["no solution"]];
Return[sols];
];

(* recover hinge direction from the centre line *)
M2B[M_]:=Module[{B},
B=Table[Cross[ M[[Mod[i-1,n,1]]],M[[Mod[i,n,1]]] ],{i,1,n+1}];
B=Table[B[[i]]/Norm[B[[i]]],{i,1,n+1}];
B[[1]]={0,0,1};
For[i=2,i<=n+1,i++,
B[[i]]=Sign[Cross[B[[i-1]],B[[i]]] . M[[Mod[i-1,n,1]]]]*B[[i]]];
Return[B];
];
(* Total Twist; does not consider slide *)
Tw[B_]:=Module[{},
If[FreeAngle || slide!=0,
Return[(1/Pi)*Total[Table[ArcCos[B[[i]] . B[[i+1]]],{i,1,n}]]],
(* Return[(n/Pi)*ArcCos[B[[2,3]]]]*)
Return[(n/Pi)*ArcCos[B[[1]] . B[[2]]]]
]
];
(* Total torsion of a curve *)
TwM[M_]:=Module[{},
Return[Tw[M2B[M]]];
];
(* dipole energy *)
Dipole[C_,P_]:=Module[{e,m,i,j,norm2},
e=0;
	For[i=1, i< Length[P], i++,
	   For[j=i+1, j<Length[P] ,j++,
		m=C[[i]]-C[[j]];norm2=m . m;
		e=e+P[[i]] . P[[j]]/norm2^(3/2)-3*(P[[i]] . m)*(P[[j]] . m)/norm2^(5/2);
];
];
	Return[e];
];
(* Print solution *)
PrintSol[sol_]:=Module[{},
Print["Cos(twist):",h[2,3]/.sol[[2]]];
Print["total local twist:",Tw[H/.sol[[2]]]];
Print["Writhe:",Writhe[M/.sol[[2]]]];
Print["Total half twist:",Writhe[M/.sol[[2]]]+TwM[M/.sol[[2]]]];Print["Hinge vectors:",Transpose[H /. sol[[2]]] // MatrixForm];
Print["Elastic Energy:", energyE /. sol[[2]]];
Print["Torsion Energy:", energyT /. sol[[2]]]; 
Print["Target func:", sol[[1]]];
];
(* display kaleidocycle *)
Disp[sol_, width_:0.5, matform_:True]:=Module[{P,centre,pnt,tri},
If[Head[sol]!=List,Return[0]];
P=H /. sol[[2]];
centre=Join[{{0,0,0}},Table[Sum[(M /. sol[[2]])[[i]],{i,m}],{m,n}]];
pnt=Flatten[Table[{centre[[i]]+width*P[[i]],centre[[i]]-width*P[[i]]},{i,n+1}],1];
tri=Flatten[Table[{{2*i+1,2*i+2,2*i+3},{2*i+1,2*i+2,2*i+4},{2*i+3,2*i+4,2*i+1},{2*i+3,2*i+4,2*i+2}},{i,0,n-1}],1];
If[oriented,Print["Dipole energy:",Dipole[centre,P]]];
If[matform,
Print["Hinge centre:",Transpose[centre] // MatrixForm];
Print["vertex position:",Transpose[pnt] // MatrixForm],
Print["vertex position:",pnt];
];
Return[Graphics3D[{GraphicsComplex[pnt,Polygon[tri]]}]];
];
(* Writhe *)
Lwrithe[u_,eps:0.001]:=Module[{n,r13,r14,r23,r24,a},
r13:=u[[3]]-u[[1]];r14:=u[[4]]-u[[1]];r23:=u[[3]]-u[[2]];r24:=u[[4]]-u[[2]];
n={Cross[r13,r14],Cross[r14,r24],
 Cross[r24,r23],Cross[r23,r13]};
a={0,0,0,0};
For[i=1,i<=4,i++,
If[Norm[n[[i]]]*Norm[n[[Mod[i+1,1,4]]]]>eps,a[[i]]=ArcSin[(n[[i]] . n[[Mod[i+1,1,4]]])/(Norm[n[[i]]]*Norm[n[[Mod[i+1,1,4]]]])]]];
Return[Sign[Cross[u[[4]]-u[[3]],u[[2]]-u[[1]]] . (u[[3]]-u[[1]])]*
Total[a]/Pi];
];
Writhe[M_]:=Module[{i,j,k,wr,U},
wr=0;U=Join[{{0,0,0}},Table[Sum[M[[k]],{k,1,i}],{i,1,n}]];
      For[j=1,j<=n-2,j++,
         For[i=j+2,i<=n,i++,
            If[i != n || j!=1,
               wr=wr+Lwrithe[{U[[i]],U[[i+1]],U[[j]],U[[j+1]]}];
];
];
];
Return[wr];
];


(* ::Input:: *)
(*(* Explicit Construction of Kaleidocycles *)*)
(*(*Basic Elliptic Theta Functions*)*)
(*Th1[x_,y_]:=EllipticTheta[1,Pi*x,Exp[-1*Pi*y]];*)
(*Th2[x_,y_]:=EllipticTheta[2,Pi*x,Exp[-1*Pi*y]];*)
(*Th3[x_,y_]:=EllipticTheta[3,Pi*x,Exp[-1*Pi*y]];*)
(*Th4[x_,y_]:=EllipticTheta[4,Pi*x,Exp[-1*Pi*y]];*)
(**)
(*(*Derivatives of Elliptic Theta Functions*)*)
(*DTh1[x_,y_]:=D[Th1[z,y],{z,1}]/. z->x;*)
(*DTh2[x_,y_]:=D[Th2[z,y],{z,1}]/. z->x;*)
(*DTh3[x_,y_]:=D[Th3[z,y],{z,1}]/. z->x;*)
(*DTh4[x_,y_]:=D[Th4[z,y],{z,1}]/. z->x;*)
(**)
(*(*Logarithmic Derivatives*)*)
(*d1[r_,y_]:=DTh1[r,y]/Th1[r,y];*)
(*d2[r_,y_]:=DTh2[r,y]/Th2[r,y];*)
(*d3[r_,y_]:=DTh3[r,y]/Th3[r,y];*)
(*d4[r_,y_]:=DTh4[r,y]/Th4[r,y];*)
(**)
(*(*Ratio Functions*)*)
(*R1[v_,r_,y_]:=Th1[-1/2*I*v+r,y]/Th1[1/2*I*v+r,y];*)
(*R3[v_,r_,y_]:=Th3[-1/2*I*v+r,y]/Th3[1/2*I*v+r,y];*)
(**)
(*(*Delta Functions*)*)
(*delta1[v_,r_,y_]:=d1[-1/2*I*v+r,y]-d1[1/2*I*v+r,y];*)
(*delta3[v_,r_,y_]:=d3[-1/2*I*v+r,y]-d3[1/2*I*v+r,y];*)
(**)
(*(*Auxiliary Functions*)*)
(*alpha2[r_,y_]:=Sqrt[Th3[2*r,y]*Th3[0,y]];*)
(*c[v_,r_,y_]:=4*Pi*I/y*1/(delta3[v,r,y]-delta1[v,r,y]);*)
(*Gam[r_,y_,b_]:=-Pi/y*(4*r+1-2*b);*)
(**)
(*(*Core f and g Functions*)*)
(*f[v_,n_,z_,r_,y_,t_]:=Th3[-1/2*I*v+r,y]*R3[v,r,y]^n*Exp[(n+1/2)*delta3[v,r,y]/(delta3[v,r,y]-delta1[v,r,y])*z+c[v,r,y]*t*z/2-Gam[r,y,1]*I*t/2]*Th2[I*v*n+z/(delta3[v,r,y]-delta1[v,r,y])+r+I*t,y];*)
(**)
(*g[v_,n_,z_,r_,y_,t_]:=Th1[1/2*I*v+r,y]*R1[v,r,y]^(-n)*Exp[(n+1/2)*delta3[v,r,y]/(delta3[v,r,y]-delta1[v,r,y])*z+c[v,r,y]*t*z/2+Gam[r,y,1]*I*t/2]*Th4[I*v*n+z/(delta3[v,r,y]-delta1[v,r,y])-r+I*t,y];*)
(**)
(*(*Special Functions G,tildeG,F,H*)*)
(*G[v_,n_,z_,r_,y_,t_]:=alpha2[r,y]*(Th1[I*v,y]/Th3[0,y])*Exp[n*delta3[v,r,y]/(delta3[v,r,y]-delta1[v,r,y])*z+c[v,r,y]*t*z/2]*R1[v,r,y]^(-n)*R3[v,r,y]^(n)*Th4[I*t+I*v*(n-1/2)+z/(delta3[v,r,y]-delta1[v,r,y]),y];*)
(**)
(*tildeG[v_,n_,z_,r_,y_,t_]:=-I*alpha2[r,y]*(Th1[I*v,y]/Th3[0,y])*Exp[n*delta3[v,r,y]/(delta3[v,r,y]-delta1[v,r,y])*z+c[v,r,y]*t*z/2]*Th4[I*t+I*v*(n-1/2)+z/(delta3[v,r,y]-delta1[v,r,y]),y];*)
(**)
(*F[v_,n_,z_,r_,y_,t_]:=alpha2[r,y]*Exp[n*delta3[v,r,y]/(delta3[v,r,y]-delta1[v,r,y])*z+c[v,r,y]*t*z/2]*Th2[I*t+I*v*(n-1/2)+z/(delta3[v,r,y]-delta1[v,r,y]),y];*)
(**)
(*H[v_,n_,z_,r_,y_,t_]:=1/alpha2[r,y]*(Th1[-1/2*I*v+r,y]*Th3[-1/2*I*v+r,y]*Th1[1/2*I*v+r,y]*Th3[1/2*I*v+r,y])*1/(Th3[2*r,y]*Th1[I*v,y])*Exp[n*delta3[v,r,y]/(delta3[v,r,y]-delta1[v,r,y])*z+c[v,r,y]*t*z/2+Gam[r,y,1]*I*t]*R1[v,r,y]^(-n)*R3[v,r,y]^(-n)*Th4[I*t+I*v*(n-1/2)+z/(delta3[v,r,y]-delta1[v,r,y])-2*r,y];*)
(**)
(*(*Coordinate Functions*)*)
(*X[v_,n_,z_,r_,y_,t_]:=(H[v,n,z,r,y,t]+Conjugate[H[v,n,z,r,y,t]])/F[v,n,z,r,y,t];*)
(**)
(*Y[v_,n_,z_,r_,y_,t_]:=1/I*(H[v,n,z,r,y,t]-Conjugate[H[v,n,z,r,y,t]])/F[v,n,z,r,y,t];*)
(**)
(*Z[v_,n_,z_,r_,y_,t_]:=n-2*D[Log[F[v,n,zt,r,y,t]],{zt,1}]/. zt->z;*)
(**)
(*(*Closing Conditions*)*)
(*close1[v_,r_,y_]:=I*y*(delta3[v,r,y]+delta1[v,r,y])+4*Pi*v;*)
(**)
(*close2[v_,r_,y_,k_,m_]:=1-Exp[m*Pi*I*(4*r+1)]*R1[v,r,y]^(-k)*R3[v,r,y]^(-k);*)
(**)
(*(*Torsion Angle Calculations*)*)
(*Tor[v_,n_,z_,r_,y_,t_]:=Log[R1[v,r,y]*R3[v,r,y]^(-1)]/I;*)
(**)
(*Tor2[v_,n_,z_,r_,y_,t_]:=Log[G[v,n-1,z,r,y,t]*Conjugate[G[v,n,z,r,y,t]]/(G[v,n,z,r,y,t]*Conjugate[G[v,n-1,z,r,y,t]])]/(2*I);*)
(**)
(*(*Curvature Angle Calculations*)*)
(*K[v_,n_,z_,r_,y_,t_]:=2*ArcTan[(Th3[0,y]/Th3[I*v,y])*(tildeG[v,n,z,r,y,t]/F[v,n,z,r,y,t])];*)
(**)
(*K2[v_,n_,z_,r_,y_,t_]:=2*ArcTan[-I*Th1[I*v,y]*Th4[I*t+I*v*(n-1/2)+z/(delta3[v,r,y]-delta1[v,r,y]),y]/(Th3[I*v,y]*Th2[I*t+I*v*(n-1/2)+z/(delta3[v,r,y]-delta1[v,r,y]),y])];*)
(**)
(*(*Binormal Vector Components*)*)
(*Bx[v_,n_,z_,r_,y_,t_]:=Sign[K[v,n,z,r,y,t]]*1/(2*I*F[v,n,z,r,y,t]*Abs[G[v,n,z,r,y,t]])*(Conjugate[g[v,n,z,r,y,t]]*f[v,n-1,z,r,y,t]-Conjugate[g[v,n-1,z,r,y,t]]*f[v,n,z,r,y,t]-Conjugate[f[v,n-1,z,r,y,t]]*g[v,n,z,r,y,t]+Conjugate[f[v,n,z,r,y,t]]*g[v,n-1,z,r,y,t]);*)
(**)
(*By[v_,n_,z_,r_,y_,t_]:=Sign[K[v,n,z,r,y,t]]*1/(2*F[v,n,z,r,y,t]*Abs[G[v,n,z,r,y,t]])*(Conjugate[g[v,n,z,r,y,t]]*f[v,n-1,z,r,y,t]-Conjugate[g[v,n-1,z,r,y,t]]*f[v,n,z,r,y,t]+Conjugate[f[v,n-1,z,r,y,t]]*g[v,n,z,r,y,t]-Conjugate[f[v,n,z,r,y,t]]*g[v,n-1,z,r,y,t]);*)
(**)
(*Bz[v_,n_,z_,r_,y_,t_]:=Sign[K[v,n,z,r,y,t]]*1/(2*I*F[v,n,z,r,y,t]*Abs[G[v,n,z,r,y,t]])*(Conjugate[f[v,n,z,r,y,t]]*f[v,n-1,z,r,y,t]-Conjugate[f[v,n-1,z,r,y,t]]*f[v,n,z,r,y,t]-Conjugate[g[v,n,z,r,y,t]]*g[v,n-1,z,r,y,t]+Conjugate[g[v,n-1,z,r,y,t]]*g[v,n,z,r,y,t]);*)
(**)
(*(*Plotting and Visualization Functions*)*)
(*(*XYZ array for curve plotting*)*)
(*XYZ[v_,z_,r_,y_,l_,t_]:=Table[{Re[X[v,k,z,r,y,t]],Re[Y[v,k,z,r,y,t]],Re[Z[v,k,z,r,y,t]]},{k,0,l}];*)
(**)
(*(*Default plot range*)*)
(*plotRange={-2.0,2.0};*)
(**)
(*(*Curve Plotting Function*)*)
(*PlotCurve[v_,z_,r_,y_,l_,t_,opts___]:=Graphics3D[{Thick,Blue,Line[XYZ[v,z,r,y,l,t]]},Axes->True,AxesLabel->{X,Y,Z},PlotRange->{plotRange,plotRange,plotRange},opts];*)
(**)
(*(*Binormal Vector Array*)*)
(*Binormal[v_,z_,r_,y_,l_,t_]:=Re[Table[{Bx[v,k,z,r,y,t],By[v,k,z,r,y,t],Bz[v,k,z,r,y,t]},{k,0,l}]];*)
(**)
(*(*Binormal Plot Function*)*)
(*BPlot[v_,z_,r_,y_,l_,t_,opts___]:=Graphics3D[{Thick,Blue,Line[Binormal[v,z,r,y,l,t]]},Axes->True,AxesLabel->{X,Y,Z},PlotRange->{plotRange,plotRange,plotRange},opts];*)
(**)
(*(*Closing Condition Plot Functions*)*)
(*PlotClose1[k_,m_]:=ContourPlot[close1[m/k*y,r,y]==0,{r,0,1},{y,0.1,2}];*)
(**)
(*PlotClose2[k_,m_]:=ContourPlot[close2[m/k*y,r,y,k,m]==0,{r,0,1},{y,0.1,2}];*)
(**)
(*PlotClosery[k_,m_]:=ContourPlot[{close1[m/k*y,r,y]==0,close2[m/k*y,r,y,k,m]==0},{r,0,1},{y,0.1,2}];*)
(**)
(*PlotClosemv[k_,r_]:=ContourPlot[{close1[v,r,v*k/m]==0,close2[v,r,v*k/m,k,m]==0},{m,0,k},{v,0.1,2}];*)
(**)
(*PlotClosemy[k_,r_]:=ContourPlot[{close1[m*y/k,r,y]==0,close2[m*y/k,r,y,k,m]==0},{m,0,k},{y,0.1,2}];*)
(**)
(*PlotClosemr[k_,y_]:=ContourPlot[{close1[m*y/k,r,y]==0,close2[m*y/k,r,y,k,m]==0},{m,0,k},{r,0.1,2}];*)
(**)


(* ::Input:: *)
(*vry = {v->0.4011699501808258`,r->.6886573386201776`,y->1.2035098523821062`, l->15};*)
(*curve=XYZ[v,0,r,y,15,0.5]/. vry;*)
(*B=Binormal[v,0,r,y,15,0.5]/. vry;*)
(*n=l /. vry;*)
(*FreeAngle=False;*)
(*slide=0;*)
(*Writhe[M]+TwM[M];*)
(**)


(* ::Input:: *)
(*XYZ[v,0,r,y,15,0.5]/. vry*)


(* ::Title:: *)
(*Kaleidocycle*)


(* ::Input:: *)
(*(* Orientable Kaleidocycle *)*)
(*Setup[7,True,0];*)


(* ::Input:: *)
(*(* maximum twist (minimising cos=h[2,3]) *)*)
(*sol=MKn[h[2,3],True];Disp[sol]*)


(* ::Input:: *)
(*(* Kn with a specified cos(twist) *)*)
(*sol=MKn[(h[2,3]-0.1)^2,True];Disp[sol,0.2]*)


(* ::Input:: *)
(*(* Kn with minimal averaged hinge *)*)
(*sol=MKn[HA . HA,True];Disp[sol]*)


(* ::Input:: *)
(*(* Non Orientable Kaleidocycle *)*)
(*Setup[8,False,0];*)


(* ::Input:: *)
(*(* minimum twist (maximising cos=h[2,3]) *)*)
(*sol=MKn[-h[2,3],True]; Disp[sol]*)


(* ::Input:: *)
(*(* minimum writhe; should be equal to maximum torsion *)*)
(*sol=MKn[Writhe[M],True]; Disp[sol]*)


(* ::Input:: *)
(*(* minimum linking number; very slow *)*)
(*Setup[8,False,0];*)
(*Tt=Writhe[M]+Tw[H];*)
(*sol=MKn[Tt]; Disp[sol]*)


(* ::Input:: *)
(*(* maximum twist with a specified linking number; very slow *)*)
(*Setup[9,True];*)
(*Tt=Writhe[M]+Tw[H];*)
(*cons=Join[cons,{Tt==4}];*)
(*sol=MKn[h[2,3],False]; *)


(* ::Input:: *)
(*(* minimum energy; this is meaningful only for the unoriented case *)*)
(*Setup[10,False,0];*)
(*sole=MKn[energyE,True];Disp[sole]*)


(* ::Input:: *)
(*(* Orientable Kn with slide  *)*)
(*Setup[7,True,2.8];*)


(* ::Input:: *)
(*sol=MKn[h[2,3],True];Disp[sol]*)


(* ::Input:: *)
(*(* minimal spring energy *) *)
(*sole=MKn[energyS,True,{"boxConstraint"->True}];Disp[sole]*)


(* ::Input:: *)
(*(* minimal bending energy *) *)
(*sole=MKn[energyE,True,{"boxConstraint"->True}];Disp[sole]*)


(* ::Input:: *)
(*(* Orientable K4 with slide: Bennett 4R ? *)*)
(*Setup[4,True,1];*)


(* ::Input:: *)
(*sol=MKn[-h[2,3]];Disp[sol,0.2]*)


(* ::Input:: *)
(*(* Non-orientable 20-fold Kaleidocycle *)*)
(*Setup[20,False,0];*)


(* ::Input:: *)
(*(* minimum twist with TT=5 *) *)
(*sol=MKn[-h[2,3],True];Disp[sol]*)


(* ::Input:: *)
(*(* Non-Orientable 7-fold Kaleidocycle with non-constant twist angle *)*)
(*Setup[7,False,0, True];sol=MKn[energyE];Disp[sol]*)


(* ::Input:: *)
(*(* Kirchhoff energy; since we demand the binormal is orthogonal to the previous tangent, we do not get constant torsion as the minimiser as opposed to the classical result on Kirchhoff rod. *)*)
(*Setup[7,True,0, True];sol=MKn[energyE+energyT];Disp[sol]*)


(* ::Input:: *)
(*(* torsion is not constant *)*)
(*HH=H /. sol[[2]];Table[HH[[i]] . HH[[i+1]],{i,1,n}]*)


(* ::Input:: *)
(*(* optimisation using initial point *)*)
(*Setup[7,True,0];*)
(*sol=MKn[(h[2,3]-0.1)^2,True];*)
(*prevSol = Transpose[Join[{vars},{vars /. sol[[2]]}]];*)
