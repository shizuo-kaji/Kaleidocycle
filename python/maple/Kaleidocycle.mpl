
;
# Function definition (automatically executed on startup)
# Startup codes
restart;
with(LinearAlgebra);
with(plots);
with(plottools);
with(Optimization);
with(combinat);
with(Student[LinearAlgebra]);
realdot := module() export `.`, norm; option builtin = 145, encrypted, cache, operator, shared, norecurse; `.` := proc(a::Vector, b::Vector) option overload; if VectorOptions(a, 'orientation') = VectorOptions(b, 'orientation') then LinearAlgebra:-DotProduct(a, b, conjugate = false); else LinearAlgebra:-Multiply(a, b); end if; end proc; norm := proc(a::Vector) local i; return sqrt(add(a[i]^2, i = 1 .. nops(a))); end proc; end module;
with(realdot);
Digits := 10;
slide := 0;
cosine := NULL;
developable := false;
variableTorsion := false;
midaxislen := 1;
oriented := false;
Gam := 0;
frames := 100;
step := 0.1;
# Utility function definition
# symbolic variables
h:=proc(i,j); return cat(h,`i`,`j`);end proc:
p:=proc(i); return cat(p,`i`);end proc:
q:=proc(i); return cat(q,`i`);end proc:

# setting up K
setupK:=proc(EndPoint::list,variableTorsion::boolean) global N,InProd,ExtProd,Nrm,Close,cn,s,Fix,c, midaxislen,oriented,developable,slide,cosine; local H,f,EP,i,m,C,T;
  unassign('c');unassign('s');
  if EndPoint=0 then
     EP:=Vector([0,0,0]);
  else
     EP:=Vector(EndPoint);
  end if;
  H:=[seq(Vector([h(i,1),h(i,2),h(i,3)]),i=1..N+1)]:
  # length and local twist constraints
  InProd:=[seq(H[i].H[i+1]-c, i=1..N)];
  Nrm:=[seq(H[i].H[i]-1,i=1..N)];
  if variableTorsion then T:=B2T(H,normalise=true) else T:=B2T(H); end if;
  # closing condition of the core: the arm end has to come back to the origin
  if developable then 
     ExtProd:=s*add((-1)^i*H[i],i=2..N+1); 
  else 
     ExtProd:=midaxislen*add(m,m=T)+s*add(H[i],i=1..N)-EP;
  end if;
  if oriented then
    Close:=[h(N+1,1)-h11,h(N+1,2)-h12,h(N+1,3)-h13];
  else
    Close:=[h(N+1,1)+h11,h(N+1,2)+h12,h(N+1,3)+h13];
  end if:
  # prepare constraint equation for Maple's optimisation routine  
  C:=[op(Nrm),op(Close),ExtProd[1],ExtProd[2],ExtProd[3]];
  if not variableTorsion then C:=[op(C),op(InProd)]; end if; 
  cn := {}: for f in C do; 
      if not f = 0 then cn := {f=0, op(cn)}; end if; 
  end do;
  Fix:=fix_global(variableTorsion);
  #print(Fix);
  if cosine <> NULL then
    Fix:={op(eval(Fix,c=cosine)),c=cosine}: ## fix cosine of torsion angle
  end if; 
  if slide <> NULL then
    Fix:={op(eval(Fix,s=slide)),s=slide}:    ## fix sliding length
  end if;
  return(H);
end proc:

# fixing global positioning
fix_global:=proc(variableTorsion) local Fix;
  Fix:=[h11=0,h12=0,h13=1,h21=0]:
  if oriented then
    Fix:=[op(Fix),h(N+1,1)=0,h(N+1,2)=0,h(N+1,3)=1];
    if not variableTorsion then Fix:=[op(Fix),h(N,3)=c,h23=c] end if;
  else
    Fix:=[op(Fix),h(N+1,1)=0,h(N+1,2)=0,h(N+1,3)=-1];
    if not variableTorsion then Fix:=[op(Fix),h(N,3)=-c,h23=c] end if;
  end if:
  return(Fix);
end proc:
infoX := proc(X1 := 0, B1 := 0) local c, k, Wr, B, T, K, X; if B1 = 0 then B := X2B(X1); else B := B1; end if; if X1 = 0 then X := B2X(B); else X := X1; end if; T := B2T(B); feasible(B, verbose = true); K := B2K(B, T, signed = true); printf("curvature %q\n", evalf(K, 4)); c := (B[1]) . (B[2]); printf("cosine %g, torsion angle %g\n", c, arccos(c)); Wr := writhe(X); printf("total K/Pi %g, total |K| %g, Wr %g, Tt %g\n", add(k, k = K)/3.141592, add(abs(k), k = K), Wr, Wr + Tw(B)); return X, B, T, K; end proc;
# clean equations
clean_eqs:=proc(eqlist) local eqs,f;
eqs:=[];
for f in eqlist do;
  if not type(lhs(f),numeric) then
    eqs:=[op(eqs),f];
  end if;
end do;
return(eqs);
end proc:
modN := proc(a) local t; global N; t := a mod N; if t = 0 then return N; else return t; end if; end proc;
signed_modN := proc(L, i) local q; global N, oriented; if oriented or 0 < i and i <= N then return L[i]; end if; q := iquo(i, N); if 0 < i then return (-1)^(q mod 2)*L[modN(i)]; else return -(-1)^(q mod 2)*L[modN(i)]; end if; end proc;
rnd2sp := proc() local a, b; a, b := evalf(1/10000000000*rand()), evalf(1/10000000000*rand()); return [cos(a)*cos(b), cos(a)*sin(b), sin(a)]; end proc;
rndips := proc() local ip, a, b, i, r; global N; a, b := evalf(1/10000000000*rand()), evalf(1/10000000000*rand()); ip := {h22 = cos(a), h23 = abs(sin(a)), cat(h, N, 1) = cos(a)*cos(b), cat(h, N, 2) = cos(a)*sin(b)}; for i from 3 to N - 1 do r := rnd2sp(); ip := {op(ip), cat(h, i, 1) = r[1], cat(h, i, 2) = r[2], cat(h, i, 3) = r[3]}; end do; return ip; end proc;
rndip := proc(eqn) local ip, x; ip := {}; for x in indets(eqn) do ip := {op(ip), x = evalf(1/1000000000000*rand() - 1)}; end do; return ip; end proc;
# find optimal Kn minimizing the given target function
find:=proc(cn, target, vars:=NULL) local found, a, sol, ip;
 found := false: 
 interface(warnlevel=0);
 while(not found) do 
 # initial point
 if vars=NULL then;
    ip:=rndips();
    else
    ip:=rndip(vars);
 end if;
 try
  found := true;
  # h22 is constrained to be positive
  a, sol := op(Minimize(target, cn, h22=-0.01..1.01, iterationlimit = 10^10, optimalitytolerance = 10^(-35), initialpoint = ip))
 catch:
  found := false;
  end try; end do; 
interface(warnlevel=4);
 return(a, sol);
end proc:
# newtonian iteration to solve equations
MPinv:=proc(A,eps:=1e-15) local v,u,s,i;
  u, s, v := SingularValues(A, output = ['U', 'S','Vt']);
# print(min(s),max(s));
  for i from 1 to Dimension(s) do;
    if s[i]>eps*max(s) then
      s[i]:=1/(s[i]);
    else
      s[i]:=0;
    end if;
  end do;
  s:=DiagonalMatrix(s,ColumnDimension(u),RowDimension(v));
  return(Transpose(MatrixMatrixMultiply(MatrixMatrixMultiply(u,s),v)));
end proc: 
newton:=proc(eqlist,vlist,J,ip0,max_iter:=2000) local fret,Jinv,ip,i,dt,a;
 ip:=ip0;
 for i from 1 to max_iter do;
   #Jinv:=MatrixInverse(eval(J,ip),method='pseudo');
   Jinv:=MPinv(eval(J,ip));
   a:=MatrixMatrixMultiply(Jinv,Vector(eval(eqlist,ip)));
   dt := min(0.1/max(abs(a)),1);  # maximum leap
   fret := Vector(eval(vlist,ip)) - dt*a;
   ip:=[seq(vlist[i] = fret[i],i=1..nops(vlist))];
   if max(map(abs,eval(eqlist,ip)))<1e-8 then
      return ip;
   end if;
 end do;
printf("newton didn't converge: res = %g\n",max(map(abs,eval(eqlist,ip))));
end proc:
# subdivision of the curve
subdiv:=proc(B) local dB,i,j,div;
 div:=2;  ## TODO:SLERP
 dB:=[seq(0,i=1..(nops(B)-1)*div+1)];
 dB[1]:=B[1];
 for i from 1 to nops(B)-1 do;
    for j from 1 to div do;
        dB[div*(i-1)+j+1]:=Normalize((1-j/div)*B[i] + (j/div)*B[i+1],Euclidean);   
    end do;
 end do;
return(dB);
end proc:
# duality
mirror:=proc(B) local i;
  return([seq( (-1)^(i-1)*B[i],i=1..nops(B))]);
end proc:
involution:=proc(B) local i;
  return([seq( Vector([(-1)^(i-1)*B[i][1],(-1)^(i)*B[i][2],(-1)^(i-1)*B[i][3]]),i=1..nops(B))]);
end proc:
negativeTwist:=proc(B) local i;
  return([seq( Vector([B[i][1],-B[i][2],B[i][3]]),i=1..nops(B))]);
end proc:
reverse:=proc(B) local i;
  return([seq(B[nops(B)+1-i],i=1..nops(B))]);
end proc:
# check feasibility for the binormals to form a Kaleidocycle
feasible:=proc(B,{verbose::boolean:=false,total::boolean:=false}) global slide; local i,m,InProd,Nrm,Closed;
    InProd:=[seq(B[i].B[i+1]-B[1].B[2], i=1..nops(B)-1)];
    Nrm:=[seq(B[i].B[i]-1,i=1..nops(B)-1)];
    Closed:=midaxislen*add(m,m=B2T(B,normalise=true))+slide*add(B[i],i=1..nops(B)-1);
    if verbose then
      printf("constant torsion %q\n", InProd);
      printf("unit norm %q\n", Nrm);
      printf("closedness %q\n", Closed);
    end if;
    if total then
       return(add(abs(i),i=InProd)+add(abs(i),i=Nrm)+add(abs(i),i=Closed));
    else
       return(InProd, Nrm, Closed);
    end if;
end proc:
closestsol := proc(U, u, istart, iend) local i, j, mins, mini, s; mins := infinity; mini := 1; for i to nops(U) do s := add(VectorNorm((U[i] - u)[j], 2, conjugate = false), j = istart .. iend); if s < mins then mins := s; mini := i; end if; end do; return mini; end proc;
dist := proc(diff) local i; return add(DotProduct(diff[i], diff[i], conjugate = false), i = 1 .. nops(diff)); end proc;
# Invariants function definition
# cosine of omega
cos_invariant:=proc(K,{mKdV::boolean:=true}) local ang,i;
   ang:=K2omega(K,':-mKdV'=mKdV);
  return(add(cos(ang[i]),i=1..N));
end proc:
# bending energy from tangent T
elastic_energy2:=proc(T,d:=2) local i;
 return add( (arccos(T[i].T[modN(i+1)]/(norm(T[i])*norm(T[modN(i+1)]))))^d ,i=1..nops(T)); 
end proc:
elastic_energy:=proc(T,d:=2) local i;
 #return add(log(1+(tan(arccos(T[i].T[modN(i+1)]/(norm(T[i])*norm(T[modN(i+1)])))/2))^2),i=1..nops(T));
 return add(log(2.0)-log(1+(T[i].T[modN(i+1)])/(norm(T[i])*norm(T[modN(i+1)]))),i=1..nops(T));
 #log(1+tan(phi/2)^2)
end proc:
torsion_energy := proc(B) local i; return add(log(2.0) - log(1 + (B[i]) . (B[i + 1])/(norm(B[i])*norm(B[i + 1]))), i = 1 .. N); end proc;
torsion_energy2 := proc(B) local i; return add(arccos((B[i]) . (B[i + 1])/(norm(B[i])*norm(B[i + 1])))^2, i = 1 .. N); end proc;
# dipole energy
dipole_energy:=proc(B,X0:=0) local X,e,i,j,m,norm2:
 if X0=0 then X:=B2X(B); else X:=X0; end if;
 e:=0;
 for i from 1 to nops(B)-1 do;
   for j from i+1 to nops(B)-1 do;
      m:=X[i]-X[j]; norm2:=m.m;
      e:=e+B[i].B[j]/norm2^(3/2)-3*(B[i].m)*(B[j].m)/norm2^(5/2);
   end do;
 end do;
 return e;
end proc:
# Coulomb potential of a curve X
coulomb_energy:=proc(X,a:=2) local e,i,j,m;
 e:=0;
 for i from 1 to nops(X)-1 do;
   for j from i+1 to nops(X)-1 do;
     m:=sqrt((X[i]-X[j]).(X[i]-X[j]));
     e:=e+1/(m^a);
   end do;
 end do;
 return e;
end proc:
curvature_recursion := proc(K) local i, s; global oriented, N; s := [seq(1, i = 1 .. N)]; if not oriented then s[1] := -1; s[nops(s)] := -1; end if; return [seq(s[i]*tan(1/2*K[modN(i + 1)])*tan(1/2*K[modN(i - 1)]) - tan(1/2*K[modN(i)])^2 + s[i]*tan(1/2*K[modN(i + 1)])*tan(1/2*K[modN(i - 1)])*tan(1/2*K[modN(i)])^2, i = 1 .. N)]; end proc;
curvature_recursion_T := proc(T) local i, s, tk; global N; tk := [seq((1 - ((T[modN(i - 1)]) . (T[modN(i)])))/(1 + ((T[modN(i - 1)]) . (T[modN(i)]))), i = 1 .. N)]; s := [seq(1, i = 1 .. N)]; if not oriented then s[1] := -1; end if; return [seq(-s[i]*sqrt(tk[modN(i + 1)]*tk[modN(i - 1)])*(1 + tk[i]) - tk[i], i = 1 .. N)]; end proc;
# local writhe by Levitt's formula
lwrithe:=proc(u) local n,r13,r14,r23,r24;
 r13:=u[3]-u[1];r14:=u[4]-u[1];r23:=u[3]-u[2];r24:=u[4]-u[2];
 n[1]:=Normalize(CrossProduct(r13,r14),Euclidean);
 n[2]:=Normalize(CrossProduct(r14,r24),Euclidean);
 n[3]:=Normalize(CrossProduct(r24,r23),Euclidean);
 n[4]:=Normalize(CrossProduct(r23,r13),Euclidean);
 return(sign(CrossProduct(u[4]-u[3],u[2]-u[1]).r13)*
    (arcsin(n[1].n[2])+arcsin(n[2].n[3])+arcsin(n[3].n[4])+arcsin(n[4].n[1])) );
end proc: ##NOTE: sign(0)=1 in Maple. This can cause a problem here.

# area of spherical polygon to compute writhe
sangle:=proc(u,v,w) local uv,vw,wu;
 uv := (u.v)/sqrt((u.u)*(v.v));vw := (v.w)/sqrt((v.v)*(w.w));wu := (w.u)/sqrt((w.w)*(u.u));
 if (1-uv^2)*(1-vw^2)=0 then return 0;end if; 
 return arccos( (wu-uv*vw)/ sqrt( (1-uv^2)*(1-vw^2) ) );
end proc:
sarea:=proc(U) local area, o, i, tarea;
  o:=Vector(rnd2sp());print(o);
  area:=0;
  for i from 1 to nops(P)-1 do;
    tarea := -Pi;
    tarea := tarea + sangle(o,U[i],U[modN(i+1)]);
    tarea := tarea + sangle(U[i],U[modN(i+1)],o);
    tarea := tarea + sangle(U[modN(i+1)],o,U[i]);
    area := area + sign(CrossProduct(o,U[i]-o).U[modN(i+1)])*tarea;
  end do;
  return evalf(area);
end proc:
# local writhe by spherical area
lwrithe2:=proc(u) local n,r13,r14,r23,r24,sg;
 r13:=u[3]-u[1];r14:=u[4]-u[1];r23:=u[3]-u[2];r24:=u[4]-u[2];
 sg:=CrossProduct(u[4]-u[3],u[2]-u[1]).(u[3]-u[1]);
 if abs(sg)<1e-5 then return 0; end if;
return(sign(sg)*
(abs(sangle(r13,r14,r24))+abs(sangle(r14,r24,r23))+abs(sangle(r24,r23,r13))+abs(sangle(r23,r13,r14))-2*Pi));
end proc:
## compute (twice the) writhe of a curve
writhe:=proc(C) local i,j,wr;
  wr:=0;
  for j from 1 to nops(C)-3 do;
    for i from j+2 to nops(C)-1 do;
        if not(i=nops(C)-1 and j=1) then
           wr:=wr+lwrithe([C[i],C[i+1],C[j],C[j+1]]);
           #wr:=wr+lwrithe2([C[i],C[i+1],C[j],C[j+1]]);
        end if;
     end do;
   end do;
  return(evalf(wr/Pi));
end proc:
# compute total twist from binormal
# from binormal
Tw:=proc(B) local i;
 return(evalf(add(arccos(B[i].B[i+1]),i=1..nops(B)-1))/Pi);
end proc:
# from curve (with slides)
TwX:=proc(X) local Bx;
 Bx:=T2B(X2T(X)):
 return(Tw(Bx));
end proc:
torsion := proc(B) local i; return [seq(arccos((B[i]) . (B[i + 1])), i = 1 .. nops(B) - 1)]; end proc;
axis := proc(B, K) local A, a1, a2, a3, rr, i; global N; A := Vector([a1, a2, a3]); rr := [seq((A . (B[i])) - tan(1/2*K[i]), i = 1 .. 3)]; return eval(A, solve(rr)); end proc;
# Conversion function definition
# compute tangent T from binormal B
B2T:=proc(B, {normalise::boolean:=false}) local T,i,v;
   T:=[seq(CrossProduct(B[i], B[i+1]), i = 1 .. nops(B)-1)];
   if normalise then
      T:=[seq(v/norm(v), v = T)];
   end if;
   return T;
end proc:
B2K := proc(B, T0 := 0, {signed::boolean := true}) local i, s, u, T, L; global oriented; if T0 = 0 then T := B2T(B, normalise = true); else T := [seq(u/norm(u), u = T0)]; end if; if signed then s := [seq(sign((CrossProduct(B[i], T[modN(i - 1)])) . (T[modN(i)])), i = 1 .. nops(T))]; else s := [seq(1, i = 1 .. nops(T))]; end if; return [seq(s[i]*arccos((T[modN(i - 1)]) . (T[modN(i)])), i = 1 .. nops(T))]; end proc;
B2K2 := proc(B) local ss, i, T; ss := sqrt(1 - ((B[1]) . (B[2]))^2); T := B2T(B, normalise = true); return [seq(arcsin(-(B[modN(i + 1)] - B[modN(i - 1)]) . (T[i])/ss), i = 1 .. N)]; end proc;
# compute the curve X from binormal (X is the origin of hinges with slide)
B2X:=proc(B,T0:=0, {barycentre::boolean:=false}) local T,X,i,u,c; global slide,midaxislen,mean;
  X := [Vector([0, 0, 0])];
  if T0=0 then T := B2T(B); else T:=[seq(u/norm(u),u=T0)]; end if;
  for i from 1 to nops(T) do;
    u:=X[-1]+T[i]*midaxislen+slide*B[i];
    X:=[op(X),u];
  end do;
  if barycentre then
    mean := add(c,c=X[..-2])/(nops(X)-1);
    X := [seq(c-mean, c = X)];
  end if;
  return X;
end proc:
X2T := proc(X, {normalise::boolean := false}) local T, i, v; T := [seq(X[i + 1] - X[i], i = 1 .. nops(X) - 1)]; if normalise then T := [seq(v/norm(v), v = T)]; end if; return T; end proc;
T2B := proc(T, Bp0 := Vector([0, 0, 1])) local B, i, b; B := [seq(CrossProduct(T[modN(i - 1)], T[modN(i)]), i = 1 .. nops(T) + 1)]; B := [seq(b/norm(b), b = B)]; B[1] := sign((B[1]) . Bp0)*B[1]; for i from 2 to nops(B) do B[i] := sign((CrossProduct(B[i - 1], B[i])) . (T[modN(i - 1)]))*B[i]; end do; return B; end proc;
X2B := proc(X, Bp0 := Vector([0, 0, 1])) return T2B(X2T(X), Bp0); end proc;
match_sgn := proc(V, W) local i; return [seq(sign((V[i]) . (W[i]))*V[i], i = 1 .. nops(V))]; end proc;
B2N := proc(B, T, {normalise::boolean := false}) local nv, v; nv := zip(CrossProduct, B, [op(T), T[1]]); if normalise then nv := [seq(v/norm(v), v = nv)]; end if; return nv; end proc;
K2B := proc(K, c) local B, T, b, t, b1, b2, b3, i, s, sol, bp, tp, R; B := [Vector([0, 0, 1])]; T := [Vector([sqrt(1 - c^2), 0, 0])]; if oriented then s := 1; else s := -1; end if; for i from 2 to nops(K) do B := [op(B), (RotationMatrix(-arccos(c), T[i - 1])) . (B[i - 1])]; T := [op(T), (RotationMatrix(K[i], B[i])) . (T[i - 1])]; end do; return [op(B), s*B[1]], T; end proc;
periodic_theta := proc(k, t := 0) local p, l; global N; p := 4/3*EllipticK(k); return [seq(2*arcsin(k*JacobiSN(l*p + t*JacobiSN(p, k)/(1 + JacobiCN(p, k)), k)), l = 1 .. N)]; end proc;
# Visualisation function definition
coords := proc(B, T, width := 0.5, offset0 := [], C0 := 0) local c, V, C, i, offset; global developable; V := []; if nops(offset0) <> nops(B) then offset := [seq(0, i = 1 .. nops(B))]; else offset := offset0; end if; if developable then c := 0.5*slide*B[1]; C := [c]; for i from 2 to nops(B) do c := c + (-1)*(-1)^i*0.5*slide*(B[i - 1] - B[i]); C := [op(C), c]; end do; else if C0 = 0 then C := B2X(B, T, ':-barycentre' = true); else C := C0; end if; end if; for i to nops(B) do V := [op(V), C[i] + (offset[i] + width)*B[i], C[i] + (offset[i] - width)*B[i]]; end do; return eval(V); end proc;
vis := proc(B, T, width := 0.5, offset := []) visTet(coords(B, T, width, offset)); end proc;
visTet := proc(crd) local i, V, k; V := map(convert, crd, list); k := 1/2*nops(V) - 2; return PLOT3D(POLYGONS(seq([V[2*i + 1], V[2*i + 2], V[2*i + 3]], i = 0 .. k), COLOR(RGB, 1.0, 0., 0.)), POLYGONS(seq([V[2*i + 1], V[2*i + 2], V[2*i + 4]], i = 0 .. k), COLOR(RGB, 1.0, 0., 0.)), POLYGONS(seq([V[2*i + 1], V[2*i + 3], V[2*i + 4]], i = 0 .. k), COLOR(RGB, 0., 0., 1.0)), POLYGONS(seq([V[2*i + 2], V[2*i + 3], V[2*i + 4]], i = 0 .. k), COLOR(RGB, 0., 0., 1.0))); end proc;
visBand := proc(crd, untwist := false) local V, k, i, j; V := map(convert, crd, list); k := 1/2*nops(V) - 2; j := floor(1/2*k); if untwist then return PLOT3D(POLYGONS(seq([V[2*i + 1], V[2*i + 2], V[2*i + 3], V[2*i + 4]], i = 0 .. k))); else return PLOT3D(POLYGONS(seq([V[2*i + 1], V[2*i + 2], V[2*i + 4], V[2*i + 3]], i = 0 .. k))); end if; end proc;
coordsTrefoil := proc(B, T, width := 0.5, barycentre := true) local M2, P2, i, offset; P2 := [op(B[1 .. -2]), -op(B[1 .. -2]), B[1]]; M2 := [op(T), op(T)]; offset := [seq(width, i = 1 .. nops(P2))]; return coords(P2, M2, 0.5*width, offset); end proc;
# visualisation as a torus
sweepTorus:=proc(B,T,width:=0.3,barycentre := false) local V,C,i,normal;
  V := []; C := B2X(B, T, barycentre); 
  for i to nops(B) do;
     normal:=CrossProduct(T[modN(i)],B[i]);
     if(i mod 4=1) then
       V := [op(V), C[i]+width*(B[i]-normal), C[i]+width*(B[i]+normal), C[i]+width*(-B[i]+normal), C[i]+width*(-B[i]-normal)];
     elif(i mod 4=2) then
       V := [op(V), C[i]+width*(-B[i]-normal), C[i]+width*(B[i]-normal), C[i]+width*(B[i]+normal), C[i]+width*(-B[i]+normal)];
     elif(i mod 4=3) then
       V := [op(V), C[i]+width*(-B[i]+normal), C[i]+width*(-B[i]-normal), C[i]+width*(B[i]-normal), C[i]+width*(B[i]+normal)];
     elif(i mod 4=0) then
       V := [op(V), C[i]+width*(B[i]+normal), C[i]+width*(-B[i]+normal), C[i]+width*(-B[i]-normal), C[i]+width*(B[i]-normal)];
     end if;
  end do;
  V := map(convert, V, list);
  return PLOT3D( 
    #seq([V[4*i+1], V[4*i+2], V[4*i+3], V[4*i+4]], i = 0 .. N-1),
    POLYGONS(seq([V[4*i+1], V[4*i+2], V[4*(i+1)+2], V[4*(i+1)+1]], i = 0 .. nops(B)-2), COLOR(RGB, 1.0, 0.0, 0.0)),
    POLYGONS(seq([V[4*i+2], V[4*i+3], V[4*(i+1)+3], V[4*(i+1)+2]], i = 0 .. nops(B)-2),COLOR(RGB, 1.0, 1.0, 0.0)), 
    POLYGONS(seq([V[4*i+3], V[4*i+4], V[4*(i+1)+4], V[4*(i+1)+3]], i = 0 .. nops(B)-2),COLOR(RGB, 0.0, 1.0, 0.0)), 
    POLYGONS(seq([V[4*i+4], V[4*i+1], V[4*(i+1)+1], V[4*(i+1)+4]], i = 0 .. nops(B)-2),COLOR(RGB, 0.0, 0.0, 1.0)));
#    seq([V[4*i+1], V[4*i+2], V[4*(i+1)+1], V[4*(i+1)+4]], i = 0 .. N-1), COLOR(RGB, 1.0, 0.0, 0.0)),
#    POLYGONS(seq([V[4*i+2], V[4*i+3], V[4*(i+1)+2], V[4*(i+1)+1]], i = 0 .. nops(P)-2), 
#    seq([V[4*i+3], V[4*i+4], V[4*(i+1)+3], V[4*(i+1)+2]], i = 0 .. nops(P)-2), 
#    seq([V[4*i+4], V[4*i+1], V[4*(i+1)+4], V[4*(i+1)+3]], i = 0 .. nops(P)-2)));
end proc:
dispCurve := proc(CM) return CURVES(map(convert, CM, list), COLOR(RGB, 0., 1.0, 0.)); end proc;
dispBd := proc(P, CM, ord := true) local i, C2; global oriented; if oriented and ord then C2 := [seq(convert(CM[i] + 1/2*P[i], list), i = 1 .. nops(P))]; else C2 := [seq(convert(CM[i] + 1/2*P[i], list), i = 1 .. nops(P) - 1), seq(convert(CM[i] - 1/2*P[i], list), i = 1 .. nops(P) - 1), convert(CM[1] + 1/2*P[1], list)]; end if; return CURVES(C2, COLOR(RGB, 0., 0., 1.0)); end proc;
apex := proc(u, v, a, b) local w, c, ca, sa; c := sqrt((u - v) . (u - v)); ca := 1/2*(a^2 + c^2 - b^2)/(a*c); sa := sqrt(1 - ca^2); w := u + a*ca*(v - u)/c + a*sa*Multiply(<0, -1 | 1, 0>, v - u)/c; return w; end proc;
tetsheet := proc(V, u1, u2) local a, b, c, d, e, f, u, i; u := [seq(i, i = 1 .. 7)]; a, b, c, d, e := sqrt((V[1] - V[2]) . (V[1] - V[2])), sqrt((V[1] - V[3]) . (V[1] - V[3])), sqrt((V[1] - V[4]) . (V[1] - V[4])), sqrt((V[2] - V[3]) . (V[2] - V[3])), sqrt((V[2] - V[4]) . (V[2] - V[4])); u[1] := u1; u[2] := u2; u[3] := apex(u[1], u[2], b, d); u[4] := apex(u[3], u[2], a, e); u[5] := apex(u[4], u[2], c, a); u[7] := apex(u[4], u[5], a, b); u[6] := apex(u[7], u[5], d, a); f := sqrt((u1 - u2) . (u1 - u2)); return u; end proc;
drawtrig := proc(u, last := false) local p; p := [polygon(map(convert, [u[1], u[2], u[3]], list))]; p := [op(p), polygon(map(convert, [u[4], u[2], u[5]], list))]; if not last then p := [op(p), polygon(map(convert, [u[3], u[2], u[4]], list))]; p := [op(p), polygon(map(convert, [u[4], u[5], u[7]], list))]; p := [op(p), polygon(map(convert, [u[7], u[5], u[6]], list))]; end if; display(p, style = line, thickness = 2, scaling = constrained, axes = none); end proc;
paper_model := proc(V) local u, pd, i; u[3] := Vector([0, 0]); u[4] := Vector([sqrt((V[1] - V[2]) . (V[1] - V[2])), 0]); pd := {}; for i to N do u := tetsheet(V, u[3], u[4]); pd := {op(pd), drawtrig(u)}; end do; u := tetsheet(V, u[3], u[4]); pd := {op(pd), drawtrig(u, true)}; return display(pd, scaling = constrained, axes = none); end proc;
# Motion function definition
## choose the sign of list elements so that they sum up to the specified value
signed_list:=proc(K,val:=0) local sgn,s,i,k;
  sgn:=choose(N):
  for s in sgn do;
     if abs(add(K[i],i=1..N)-2*add(K[i],i=s)-val)<0.001 then
        k:=[];
        for i from 1 to N do;
          if i in s then k:=[op(k),-K[i]] else k:=[op(k),K[i]] end if;
        end do;
        return(k); 
     end if;
  end do: 
  return(NULL);
end proc:
# compute the angle (denoted by omega in the paper) for deformation by sine-Gordon and mKdV
K2omega:=proc(K,{mKdV::boolean:=false}) local ss,s,phi,p,i,j,u,ang; global oriented,N;
  if oriented then s:=1 else s:=-1; end if;
  if mKdV then # -phi[i-1]-phi[i]=K[i]
     ss:=solve([seq(phi[i+1]+phi[i]+K[i+1]=0,i=1..N-1), phi[1]+s*phi[N]+K[1]=0]);
  else #sine-Gordon
     if oriented then;
       phi:=[p];
       for i from 2 to N do;
          phi:=[op(phi),phi[i-1]-K[i]]; # phi[i-1]-phi[i]=K[i]
       end do;
       ss:=[p=fsolve(add(sin(u),u in phi)=0)]; # enforcing the presevation of the cos invariants
     else
       #ss:=solve([seq(phi[i+1]-phi[i]+K[i+1]=0,i=1..N-1), phi[1]-s*phi[N]+K[1]=0]);
       phi:=[seq(0.5*(add(K[i], i = j + 1 .. N) - add(K[i], i = 1 .. j)), j = 1 .. N)]; # equivalent
       return([op(phi),-phi[1]]);
     end if;
  end if;
  if nops(ss)=0 then
     print("no solution for omega");
     ang:=[seq(0,i=1..N)];
  else:
     ang:=eval([seq(phi[i],i=1..N)],ss);
  end if;
  return([op(ang),s*ang[1]]);
end proc:
##
omega2K:=proc(phi,{mKdV::boolean:=false}) local s,u,i; global oriented,N;
  if oriented then s:=1 else s:=-1; end if;
  if mKdV then u:=1 else u:=-1; end if;
  return([-phi[1]-u*s*phi[N],seq(-phi[i]-u*phi[i-1],i=2..N)]);
end proc:
K2theta := proc(K) local ss, phi, i, ang, sysc; global N; sysc := [seq(signed_modN(phi, i + 1) - signed_modN(phi, i - 1) - 2*K[i] = 0, i = 1 .. N)]; ss := solve(sysc, [seq(phi[i], i = 1 .. N)]); if nops(ss) = 0 then print("no solution for theta"); else ang := eval([seq(phi[i], i = 1 .. N)], ss[1]); end if; return ang; end proc;
theta2K := proc(phi) local i, Kr; global N; Kr := [seq(1/2*signed_modN(phi, i + 1) - 1/2*signed_modN(phi, i - 1), i = 1 .. N)]; return Kr; end proc;
diffX := proc(X, Bp0, {rule := "sine-Gordon-X", B0 := 0}) local T, B, K, U, i, ang, c; global oriented; if B0 = 0 then T := X2T(X, normalise = true); B := T2B(T, Bp0); else B := B0; T := B2T(B, normalise = true); end if; K := B2K(B, T, singed = true); U := B2N(B, T, normalise = true); if rule = "mKdV2-X" then c := (B[1]) . (B[2]); return [seq(c*T[modN(i)] - c*tan(0.5*K[modN(i)])*U[modN(i)] + sqrt(1 - c^2)*tan(0.5*K[modN(i)])*B[modN(i)], i = 1 .. nops(X))]; else ang := K2omega(K, ':-mKdV' = eval(rule = "mKdV-X")); return [seq(sin(ang[i])*U[i] + cos(ang[i])*T[modN(i)], i = 1 .. nops(X))]; end if; end proc;
sGmoveX := proc(X, Bp0, h := 0.01, {rule := "sine-Gordon-X"}) local k1, k2, k3, k4; k1 := diffX(X, Bp0, 'rule' = rule); k2 := diffX(eval(X + 0.5*h*k1), Bp0, ':-rule' = rule); k3 := diffX(eval(X + 0.5*h*k2), Bp0, ':-rule' = rule); k4 := diffX(eval(X + h*k3), Bp0, ':-rule' = rule); return eval(X + 1/6*h*(k1 + 2*k2 + 2*k3 + k4)); end proc;
diffB := proc(B, {rule := "sin-Gordon"}) local P, T, U, K, i, ang, p, q, s, c, p2q2; global oriented; T := B2T(B, normalise = true); K := B2K(B, T, signed = true); U := B2N(B, T, normalise = true); if rule = "mKdV2" then if oriented then s := 1; else s := -1; end if; P := [seq(-tan(1/2*K[i])*T[i] - U[i], i = 1 .. nops(T))]; return [op(P), s*P[1]]; else ang := K2omega(K, ':-mKdV' = eval(rule = "mKdV")); return [seq(-cos(ang[i])*U[i] + sin(ang[i])*T[modN(i)], i = 1 .. nops(B))]; end if; end proc;
sGmoveB := proc(B, h := 0.01, {rule := "sine-Gordon"}) local i, P, k1, k2, k3, k4; k1 := diffB(B, ':-rule' = rule); k2 := diffB(eval(B + 0.5*h*k1), ':-rule' = rule); k3 := diffB(eval(B + 0.5*h*k2), ':-rule' = rule); k4 := diffB(eval(B + h*k3), ':-rule' = rule); P := eval(B + 1/6*h*(k1 + 2*k2 + 2*k3 + k4)); return [seq(P[i]/norm(P[i]), i = 1 .. nops(P))]; end proc;
project_sol := proc(B, c0) local res, sol, P; global cn; res, sol := find([op(cn), c = c0], dist(H, B)); P := eval(H, sol); printf("dist %g, residue %g\n", dist(P, B), res); return P; end proc;
# Animation function definition
# compute solutions step by step
anim_step:=proc(sol, st:=1e-3) local acn,i,S,s,a,soln,prev,c0,Hfc,step,J,eqlist,vlist;global frames,N,cn_fixed;
  c0:= eval(c, sol);
  Hfc:=eval(Hf,c=c0);
  acn := map(lhs,[op(eval(cn_fixed,c=c0))]):
  vlist:=[op(indets(acn))];
  S := [eval(Hfc,sol)]:
  step:=st*N;
  soln:=sol;
for i from 2 to frames do;
   if i=2 then
     s := dist(eval(S[1]-Hfc)[2..N]):
     eqlist:=[op(acn),s-step];
   else
     s := dist(eval(S[-2]-Hfc)[2..N]):
     eqlist:=[op(acn),s-2*step];
   end if;
 #  printf("%g / %g\n",i,frames);
   J:=VectorCalculus[Jacobian](eqlist,vlist);
   soln:=newton(eqlist,vlist,J,soln);
   S:=[op(S),eval(Hfc,soln)];
end do:
return(S);
end proc:
# compute random solutions
anim_rnd:=proc(B) local i,S,a,soln; global frames;
  S := [B]:  
  for i from 2 to frames do;
   #  printf("%g / %g\n",i,frames);
     a, soln := find(cn_fixed,target);
     S:=[op(S),eval(Hf,soln)];
  end do:
return(S);
end proc:
# animation by sine-Gordon and mKdV
anim_dd:=proc(B,step:=0.02,{rule:="sine=Gordon"}) local s,S,i,j;
  S := [B]:
  for i from 2 to frames do;
    s:=sGmoveB(S[-1],step,':-rule'=rule);
    S:=[op(S),s];
  end do:
return(S);
end proc:
#
anim_ddX:=proc(B,step:=0.02,{rule:="sine-Gordon-X"}) local s,S,i,X;
  S := [B]:
  for i from 2 to frames do;
    X:=B2X(S[-1]);
    s:=sGmoveX(X,S[-1][1],step,':-rule'=rule);
    S:=[op(S),T2B(X2T(s),S[-1][1])];
  end do:
return(S);
end proc:
anim_load := proc(filename) local Z, S, i, j; global frames, N; Z := ImportMatrix(filename, source = csv); frames := RowDimension(Z); N := 1/3*Dimension(Z[1]) - 1; Z := ArrayTools[Reshape](Z, [frames, 3, N + 1]); S := [seq([seq(Vector[column](Z[i][1 .. -1, j]), j = 1 .. N + 1)], i = 1 .. frames)]; return S; end proc;
#clean solutions by removing or negating wrong twist
clean_anim:=proc(S,eps:=0.01,{twistFix:=true,Xs:=0}) local i,q, Bs, Ts, Ks, Vs, C; global width, variableTorsion;
  Bs := []: Vs:=[]: Ts:=[]:Ks:=[]:
  for i from 1 to nops(S) do;
  if (S[i][1].S[i][2]-S[1][1].S[1][2])^2 >eps or feasible(S[i],total=true)>eps then
     printf("%g,",i);
  else
     if (S[i][2][2]<0) and twistFix then
        Bs:=[op(Bs),negativeTwist(S[i])];
     else
        Bs:=[op(Bs),S[i]];
     end if;
     if variableTorsion then
       Ts := [op(Ts),B2T(Bs[-1],normalise=true)];
     else
       Ts := [op(Ts),B2T(Bs[-1])];
     end if;
     if X0=0 then C:=0 else C:=Xs[i] end if;
     Vs := [op(Vs), coords(Bs[-1],Ts[-1],width,[],C)];
     Ks := [op(Ks), B2K(Bs[-1], Ts[-1])];
  end if:
  end do:
  printf("%g frames after cleaning %g frames", nops(Bs), nops(S));
return(Bs, Ts, Ks, Vs);
end proc:
# sort solution in terms of the location of the Nth hinge
sort_anim:=proc() local i,q,Bs0,Ts0,Vs0; global Bs,Ts,Vs;
  q:=Vs[1]: Bs:=[Bs[1]]: Ts:=[Ts[1]]: Vs:=[q]:
  Bs0:=subsop(1=NULL,Bs):Vs0:=subsop(1=NULL,Vs): Ts0:=subsop(1=NULL,Ts);
  while(nops(Vs0)>0) do;
     i:=closestsol(Vs0,q,N-2,N+1);
     Bs:=[op(Bs),Bs0[i]]; Ts:=[op(Ts),Ts0[i]];
     q:=Vs0[i]; Vs:=[op(Vs),q];
     Bs0:=subsop(i=NULL,Bs0);Ts0:=subsop(i=NULL,Ts0);
     Vs0:=subsop(i=NULL,Vs0);
  end do: 
  return(nops(Vs));
end proc:
fix_rot := proc(Bs1, Ts1) local Xs, R, u, s, v, i, A, Bs, Ts, Vs; Xs := [seq(Matrix(B2X(Bs1[i], Ts1[i], barycentre = true)[1 .. N]), i = 1 .. nops(Bs1))]; R := []; for i to nops(Bs1) do A := Multiply(Xs[1], Transpose(Xs[i])); u, s, v := SingularValues(A, output = ['U', 'S', 'Vt']); if Determinant(A) < 0 then u := Multiply(u, DiagonalMatrix([1, 1, -1])); end if; R := [op(R), Multiply(u, v)]; end do; Bs := [seq(map2(Multiply, R[i], Bs1[i]), i = 1 .. nops(Bs1))]; Ts := map(B2T, Bs1); Vs := [seq(coords(Bs[i], Ts[i], width), i = 1 .. nops(Bs))]; return Bs, Ts, Vs; end proc;
disp_anim := proc(Vs, func) local i, v; v := [seq(func(Vs[i]), i = 1 .. nops(Vs))]; display(v, insequence = true, scaling = constrained, axes = none); end proc;
# Tau function definition
NULL;
th1 := (x, y) -> JacobiTheta1(Pi*x, exp(-Pi*y));
dth1 := (x, y) -> subs(z = x, diff(th1(z, y), z));
d1 := (r, y) -> dth1(r, y)/th1(r, y);
th2 := (x, y) -> JacobiTheta2(Pi*x, exp(-Pi*y));
dth2 := (x, y) -> subs(z = x, diff(th2(z, y), z));
d2 := (r, y) -> dth2(r, y)/th2(r, y);
th3 := (x, y) -> JacobiTheta3(Pi*x, exp(-Pi*y));
dth3 := (x, y) -> subs(z = x, diff(th3(z, y), z));
d3 := (r, y) -> dth3(r, y)/th3(r, y);
th4 := (x, y) -> JacobiTheta4(Pi*x, exp(-Pi*y));
dth4 := (x, y) -> subs(z = x, diff(th4(z, y), z));
d4 := (r, y) -> dth4(r, y)/th4(r, y);
R1 := (v, r, y) -> th1(-1/2*I*v + r, y)/th1(1/2*I*v + r, y);
R3 := (v, r, y) -> th3(-1/2*I*v + r, y)/th3(1/2*I*v + r, y);
delta1 := (v, r, y) -> d1(-1/2*I*v + r, y) - d1(1/2*I*v + r, y);
delta3 := (v, r, y) -> d3(-1/2*I*v + r, y) - d3(1/2*I*v + r, y);
alpha2 := (r, y) -> sqrt(th3(2*r, y)*th3(0, y));
beta2 := (r, y) -> th3(v*I, y)/th3(0, y);
eC := (v, r, y) -> 4*I*Pi/(y*(delta3(v, r, y) - delta1(v, r, y)));
eG := (v, z, r, y, j, t) -> alpha2(r, y)*th1(v*I, y)*exp(j*delta3(v, r, y)*z/(delta3(v, r, y) - delta1(v, r, y)) + 1/2*eC(v, r, y)*t*z)*R1(v, r, y)^(-j)*R3(v, r, y)^j*th4(t*I + v*(j - 1/2)*I + z/(delta3(v, r, y) - delta1(v, r, y)), y)/th3(0, y);
eF := (v, z, r, y, j, t) -> alpha2(r, y)*exp(j*delta3(v, r, y)*z/(delta3(v, r, y) - delta1(v, r, y)) + 1/2*eC(v, r, y)*t*z)*th2(t*I + v*(j - 1/2)*I + z/(delta3(v, r, y) - delta1(v, r, y)), y);
eH := (v, z, r, y, j, t) -> th1(-1/2*I*v + r, y)*th3(-1/2*I*v + r, y)*th1(1/2*I*v + r, y)*th3(1/2*I*v + r, y)*exp(j*delta3(v, r, y)*z/(delta3(v, r, y) - delta1(v, r, y)) + 1/2*eC(v, r, y)*t*z + Gam*t*I)*R1(v, r, y)^(-j)*R3(v, r, y)^(-j)*th4(t*I + v*(j - 1/2)*I + z/(delta3(v, r, y) - delta1(v, r, y)) - 2*r, y)/(alpha2(r, y)*th3(2*r, y)*th1(v*I, y));
eX := (v, z, r, y, j, t) -> 2*Re(eH(v, z, r, y, j, t))/eF(v, z, r, y, j, t);
eY := (v, z, r, y, j, t) -> 2*Im(eH(v, z, r, y, j, t))/eF(v, z, r, y, j, t);
eZ := (v, z, r, y, j, t) -> j - 2*subs(zz = z, diff(log(eF(v, zz, r, y, j, t)), zz));
XYZ := proc(v, z, r, y, N, t) local j; return [seq(Vector([Re(evalf(eX(v, z, r, y, j, t))), Re(evalf(eY(v, z, r, y, j, t))), Re(evalf(eZ(v, z, r, y, j, t)))]), j = 0 .. N)]; end proc;
eTor := (v, z, r, y, t) -> Re(-I*log(R1(v, r, y)/R3(v, r, y)));
eK := (v, z, r, y, j, t) -> Re(2*arctan(-I*th1(v*I, y)*th4(t*I + v*(j - 1/2)*I + z/(delta3(v, r, y) - delta1(v, r, y)), y)/(th3(v*I, y)*th2(t*I + v*(j - 1/2)*I + z/(delta3(v, r, y) - delta1(v, r, y)), y))));
close1 := (v, r, y) -> y*(delta3(v, r, y) + delta1(v, r, y))*I + 4*Pi*v;
close2 := (v, r, y, N) -> 1 - exp(N*v*Pi*(4*r - 1)*I/y)*R1(v, r, y)^(-N)*R3(v, r, y)^(-N);
# Start here: Find Kaleidocycles
# set global constants
Digits:=10; # numerical accuracy
midaxislen:=1: # length of each segment
N:=9; # number of tetrahedra
oriented := true; # true: oriented, false: non-oriented Kaleidocycles

## hinge vectors
H:=setupK([0,0,0],variableTorsion):
cn_fixed := clean_eqs(eval(cn,Fix)):
Hf:=eval(H,Fix);HA:=add(H[i],i=1..N): 
### set the objective function to be minimised when finding a particular Kaleidocycle
### choose one of them by uncommenting the corresponding line
## local twist (torsion)
if oriented then
  target:=c     # for N: odd, c \in [-a,1]; for N: even, c \in [-1,1]
else
  target:=-c    # for N: odd, c \in [-1,a]; for N: even, c \in [-a,a]
end if:
## slide
#target := s;
## the length of the averaged hinge vector
#target:=HA.HA:
## writhe
#target:=writhe(B2T(Hf)):
## torsional energy; meaningful only for the unoriented case
#target:=elastic_energy(B2T(Hf)):
## Kirchhoff energy: currently, we do not know any connection to Kaleidocycle
#target:=0.5*torsion_energy(Hf)+elastic_energy(B2T(Hf)):
#target:=torsion_energy(Hf):
## dipole energy
#target:=dipole_energy(H,B2X(Hf,B2T(Hf))):
## fixed twist 
#target:= (c+0.1)^2;
## curvature recursion formula (does not work)
#rc := curvature_recursion_T(B2T(Hf, normalise = true)):
#rc:=curvature_recursion(B2K2(Hf)):
#target:=add((rc[i] - rc[i + 1])^2, i = 1 .. N - 1):
T := B2T(H, normalise = true);
unassign('q');
L := Vector([0, 0, q]);
cn := eval(cn, [s = 0]);
target := add(((1 + ((T[modN(i - 1)]) . (T[modN(i)])))*(L . (H[i]))^2 + (-1 + ((T[modN(i - 1)]) . (T[modN(i)]))))^2, i = 1 .. N);
a, sol := find(cn, target, [op(cn), q]);
B := eval(H, sol);
print(a, sol[1]);
s := [seq(1, i = 0 .. N + 1)];
if not oriented then
    s[1] := -1;
    s[nops(s)] := -1;
end if;
unassign('tk');
u1 := [seq(s[i]*tk[modN(i + 1)]*tk[modN(i - 1)]*(tk[i]^2 + 1) - tk[i]^2, i = 1 .. N)];
u2 := [seq(((s[i]*tk[modN(i + 1)] + tk[modN(i - 1)])*(tk[i]^2 + 1))/(2*tk[i]), i = 1 .. N)];
creq := [seq(u1[i] - u, i = 1 .. N)];
cl := expand(add((tk[modN(i - 1)] + tk[modN(i + 1)])/(2*tk[i]) + tk[i]/2*(tk[modN(i - 1)] - tk[modN(i + 1)]) - c, i = 1 .. N));
eqn := [op(creq), cl, u + 1.488436964];
sol := fsolve(eqn, indets(eqn));
ev := [u = -1.488436964];
target := add(eval(creq[i], ev)^2, i = 1 .. N);
a, sol := op(Minimize(target, iterationlimit = 10^13, optimalitytolerance = 10^(-35)));
eval([target, cl], [seq(tk[i] = uu[i], i = 1 .. N)]);
eval(target, sol);
# find a Kaleidocycle which minimises "target" under constraints
a, sol := find(cn_fixed,target): 
printf("target function value %a",a);
#slide:=eval(s,sol);
B:=eval(Hf,sol):
# print solution
T:=B2T(B,normalise=true): X:=B2X(B,T,':-barycentre'=true):print("hinges",evalf(B,4));
if variableTorsion then printf("torsion %q\n",evalf(torsion(B),4));c:=h23; end if:
costor:=eval(c,sol):tor:=arccos(costor):
printf("cos %g, radian %g, degree %g, total angle(rad) %g\n",costor,tor,evalf(180*tor/Pi), N*tor);
printf("bending energy %g, torsional energy %g, dipole energy %g, Coulomb energy %g\n",elastic_energy(T),torsion_energy(B), dipole_energy(B,X),coulomb_energy(X,2));
K:=B2K(B,T,signed=true):
printf("curvature %q\n",evalf(K,4));
printf("cos invariant %g ",cos_invariant(K,':-mKdV'=false));
#evalf(sarea(T)/Pi-2);
Wr:=Re(writhe(X)):
printf("total K/Pi %g, total |K| %g, Wr %g, Tw %g, Tt %g\n",add(k, k = K)/3.141592,add(abs(k), k = K), Wr,Tw(B),Wr+TwX(X));
print("hinge average",add(B[i],i=1..N)/N);
L := Vector([ax, ay, az]);
tg := [seq((L . (B[i])) - tan(K[i]/2), i = 1 .. N)];
AA := eval(L, fsolve([tg[1], tg[2], tg[3]]));
-2*tan(tor)*sqrt(AA . AA);
seq(b . AA, b in B);
seq((AA . (B[i])) - tan(K[i]/2), i = 1 .. N);
a1 := seq(AA . (T[i] + T[modN(i + 1)])/sin(tor), i = 1 .. N);
a2 := seq(-2*costor + ((sin(tor)*AA) . (T[i] + T[modN(i + 1)])), i = 1 .. N);
uu := [seq(tan(K[i]/2), i = 1 .. N)];
eval(u1, [seq(tk[i] = uu[i], i = 1 .. N)]);
eval(u2, [seq(tk[i] = uu[i], i = 1 .. N)]);
c3 := 2*u2[2]/costor - 2;
c1 := sqrt(-(2*u1[2]/costor^2)*2 + c3^2);
tc3 := c3 + c1^2/4;
tc1 := c1*sqrt(1 + c3/2 + c1^2/16);
eval(c3, [seq(tk[i] = uu[i], i = 1 .. N)]);
eval(c1, [seq(tk[i] = uu[i], i = 1 .. N)]);
eval(tc3/tc1, [seq(tk[i] = uu[i], i = 1 .. N)]);
curvature_recursion(K);
feasible(B, verbose = true);
width := 0.8;
V := coords(B, T, width);
visTet(V);
display(visBand(V), axes = none);
sweepTorus(B, T, 0.5*width);
# mirror duality
Bm:=mirror(B):
#Bm:=involution(B):
Tm:=B2T(Bm):Vm:=coords(Bm,Tm,0.3):Xm:=B2X(Bm, Tm):
#B;Bm;T;Tm;feasible(Bm, verbose = false);
Wr:=writhe(Xm):printf("Wr = %g, Tt=%g\n",Wr,Wr+Tw(Bm));
printf("curvature of original %q\n",evalf(K,4));printf("curvature of mirrored %q\n",evalf(B2K(Bm,Tm,signed=true),4));
display(dispBd(Bm, Xm, false), dispCurve(Xm), display(visBand(coords(Bm, Tm)), transparency = .8), axes = none);
Bm := negativeTwist(B);
Tm := B2T(Bm);
Vm := coords(Bm, Tm, 0.3);
Xm := B2X(Bm, Tm);
Wr := writhe(Xm);
printf("Wr = %g, Tt=%g\n", Wr, Wr + Tw(Bm));
printf("curvature of original  %q\n", evalf(K, 4));
printf("curvature of negative %q\n", evalf(B2K(Bm, Tm, signed = true), 4));
display(dispBd(Bm, Xm, false), dispCurve(Xm), display(visBand(coords(Bm, Tm)), transparency = 0.8), axes = none);
# Various Visualisation
offset := [seq(slide/2, i = 1 .. nops(B))];
V := coords(B, T, 0.6, offset);
visBand(V);
visBand(coordsTrefoil(B, T, 0.5));
width := 0.5;
BB := width*B;
display(dispBd(BB, X, true), dispCurve(X), display(visBand(coords(BB, T), false), transparency = 0.5), axes = none);
display(dispCurve(X), dispBd(B, X));
# Global optimization (requires Global Optimization Toolbox)
with(GlobalOptimization);
infolevel[GlobalOptimization] := 3;
a, sol := op(GlobalSolve(target, cn_fixed, seq(a = -1 .. 1, a in indets(cn_fixed)), initialpoint = rndips(), avgstopstepwidth = 0.1e-7));
# Animation
NULL;
frames:=100; width:=0.8:chiral:=false:
# choose how the evolution is generated by uncommenting one line from below
#evolution:="mKdV";     ## mKdV flow
#evolution:="sine-Gordon"; ## sine-Gordon flow
evolution:="mKdV2"; ## another mKdV flow
#evolution:="neighbour";  ## neighbour search by newtonian iteration
#evolution:="random";  ## random configurations
#evolution:="matlab/sols_move.csv"; ## load from file
## flow w.r.t. vertex coordinates becomes unstable near K[i]=0
#evolution:="sine-Gordon-X"; ## sine-Gordon flow for vertex coordinates
#evolution:="mKdV2-X"; ## matsuura's 

;
if evolution = "neighbour" then S := anim_step(sol, 0.002); elif evolution = "random" then S := anim_rnd(B); sort_anim(); elif evolution = "mKdV" or evolution = "mKdV2" or evolution = "sine-Gordon" then S := anim_dd(B, 0.05, ':-rule' = evolution); elif evolution = "mKdV-X" or evolution = "mKdV2-X" or evolution = "sine-Gordon-X" then S := anim_ddX(B, 0.05, ':-rule' = evolution); else S := anim_load(evolution); slide := 0; end if;
if chiral then
    S := Map(mirror, S);
end if;
width := 0.5;
midaxislen := 1;
eps := 0.01;
Bs, Ts, Ks, Vs := clean_anim(S, eps, twistFix = false);
Bs, Ts, Vs := fix_rot(Bs, Ts);
disp_anim(Vs, visBand);
w := [seq(convert(Vs[i][N], list), i = 1 .. nops(Vs))];
pointplot3d(w);
disp_anim(Vs, visBand);
display([seq(visBand(coordsTrefoil(Bs[i], Ts[i], 0.3, true)), i = 1 .. nops(Bs))], insequence = true, scaling = constrained, axes = none);
display([seq(sweepTorus(Bs[i], Ts[i], width, true), i = 1 .. nops(Bs))], insequence = true, scaling = constrained, axes = none);
Xs := [seq(B2X(Bs[i], Ts[i], barycentre = true), i = 1 .. nops(Bs))];
display([seq(display(dispCurve(Xs[i]), display(visBand(Vs[i]), transparency = 0.9), display(dispBd(Bs[i], Xs[i], false), color = green, transparency = 0.9)), i = 1 .. nops(Bs))], style = line, thickness = 3, insequence = true, axes = none);
display([seq(display(dispCurve(Xs[i])), i = 1 .. nops(Bs))], style = line, thickness = 3, insequence = false, axes = none);
Xs := [B2X(S[i])];
P := [S[1]];
for i to frames do
    Xs := [op(Xs), sGmoveX(Xs[-1], P[-1][1])];
    P := [op(P), T2B(X2T(Xs[-1]), P[-1][1])];
end do;

# Plot various quantities
m := [seq(display(dispCurve(Bs[i]), plottools[sphere]([0, 0, 0], 1.0, transparency = 0.8)), i = 1 .. nops(Bs))];
display(m, style = line, thickness = 3, insequence = true, scaling = constrained, axes = none);
m := [seq(display(dispCurve([op(Ts[i]), Ts[i][1]]), plottools[sphere]([0, 0, 0], sqrt((Ts[i][1]) . (Ts[i][1])), transparency = 0.8)), i = 1 .. nops(Ts))];
display(m, style = line, thickness = 3, insequence = true, scaling = constrained, axes = none);
if oriented then s := 1; else s := -1; end if;
plot([seq([i, add(s^k*Ks[i][k], k = 1 .. nops(Ks[i]))], i = 1 .. nops(Ks))]);
display([seq(listplot(k), k = Ks)], style = line, thickness = 3, insequence = true);
plot([seq([i, cos_invariant(Ks[i], ':-mKdV' = true)], i = 1 .. nops(Bs))]);
plot([seq([i, feasible(Bs[i], total = true)], i = 1 .. nops(Bs))]);
u := [seq([i, elastic_energy(Ts[i])], i = 1 .. nops(Ts))];
plot(u);
u := [seq([i, elastic_energy2(Ts[i])], i = 1 .. nops(Ts))];
plot(u);
u := [seq([i, torsion_energy(Bs[i])], i = 1 .. nops(Bs))];
plot(u);
Xs := [seq(B2X(Bs[i], Ts[i]), i = 1 .. nops(Bs))];
plot([seq([i, dipole_energy(Bs[i], Xs[i])], i = 1 .. nops(Bs))]);
plot([seq([i, coulomb_energy(Xs[i], 2)], i = 1 .. nops(Bs))]);
# plot averaged hinge length
plot([seq([i, Norm(add(Bs[i][j],j=1..nops(Bs[i])-1),Euclidean)], i = 1 .. nops(Bs))]);
# Explicit solution
NULL;
Gam := 0;
frames := 100;
step := 0.1;
NULL;
vals := [v = 0.4866323328934475, r = 0.27324332614523567, y = 1.1354754461003873];
N := 7;
vals := [v = 0.3999817876154305, r = 0.7166217679947888, y = 1.0666181013377332];
N := 8;
vals := [v = 0.6666363120897519, r = 0.7833782318210762, y = 1.0666181003877289];
N := 8;
vals := [v = 0.34219436436572975, r = 0.28899297783523453, y = 1.0265830971482972];
N := 9;
vals := [v = 0.414589, r = 0.260301, y = 0.932826];
N := 9;
vals := [v = 0.4011699501808258, r = 0.6886573386201776, y = 1.2035098523821062];
N := 15;
vals := [v = 0.189406371136433, r = 0.2995618958816959, y = 0.9470318556821652];
N := 15;
vals := [v = 0.07227972073349694, r = 0.30353311936556515, y = 0.9155431292909612];
N := 38;
m := round(eval(v*N/y, vals));
eval(close1(v, r, y), vals);
eval(close2(v, r, y, N), vals);
oriented := false;
print("oriented:", oriented);
X := eval(XYZ(v, 0, r, y, N, 0.05), vals);
B := X2B(X);
K := B2K(B);
infoX(0, B);
curvature_recursion(K);
Bi := involution(B);
add(Bi[i], i = 1 .. N);
axis(B, K);
sin(arccos(0.853305));
display([seq(listplot([seq(eval(eK(v, 0, r, y, j, step*t), vals), j = 0 .. N)]), t = 0 .. frames)], style = line, thickness = 3, insequence = true);
torang := evalf(eval(eTor(v, 0, r, y, 0.01), vals));
cos(torang);
H := setupK([0, 0, 0], false);
ip := {seq(seq(cat(h, i, j) = B[i][j], i = 1 .. N + 1), j = 1 .. 3)};
a, sol := op(Minimize(-c, eval(cn, s = 0), h22 = -0.01 .. 1.01, iterationlimit = 10^10, optimalitytolerance = 10^(-10), initialpoint = ip));
eval(H, ip);
eval(H, sol);
width := 0.3;
midaxislen := 1;
step := 0.05;
frames := 50;
Xs := [Threads:-Seq(eval(XYZ(v, 0, r, y, N, step*t), vals), t = 1 .. frames)];
Bs := [X2B(Xs[1])];
for x in Xs[2 .. -1] do
    Bs := [op(Bs), X2B(x, Bs[-1][1])];
end do;
eps := 0.01;
Bs, Ts, Ks, Vs := clean_anim(Bs, eps, twistFix = false, ':-Xs' = Xs);
Bs, Ts, Vs := fix_rot(Bs, Ts);
disp_anim(Vs, visTet);
# experimental
with(GlobalOptimization);
infolevel[GlobalOptimization] := 3;
infolevel[Optimization] := 0;
N := 7;
closedness := close2(v, r, y, N)*conjugate(close2(v, r, y, N));
Minimize(closedness, v = 0.1 .. 1, r = 0.1 .. 1, y = 0.1 .. 1.5, assume = nonnegative);
K1 := [seq(arcsin(k*JacobiCN(3*EllipticK(k)*l, k)), l = 1 .. N)];
listplot(K1);
a;
k := fsolve(arccos(a)^2 = 1/(4*z)*(2*EllipticE(z)/EllipticK(z) - 1));
k := 0.9;
Vs := [];
for t from 0 to frames do
    B1, T1 := K2B(Theta2K(periodic_theta(k, step*t)), a);
    Vs := [op(Vs), coords(B1, T1, width)];
end do;
display([seq(visTet(V), V = Vs)], insequence = true, axes = none);
# Paper model
# first, compute the vertices (if you already computed one in some sections above, you can skip)`
N:=10; # number of tetrahedra
oriented := false; # this must be set to false
H:=setupK([0,0,0],variableTorsion):
cn_fixed := clean_eqs(eval(cn,Fix)):
Hf:=eval(H,Fix):
a, sol := find(cn_fixed,-c): 
printf("cosine of torsion angle %a",a);
B:=eval(Hf,sol):T:=B2T(B,normalise=true): X:=B2X(B,T,':-barycentre'=true):
width := 0.5;
V := coords(B, T, width);
paper_model(V);
# Interacting with Mathematica
filename := "K15.csv";
unassign('c');
X := ImportMatrix(filename, source = csv);
N := RowDimension(X) - 1;
X := [seq(Vector[column](X[j, 1 .. -1]), j = 1 .. N + 1)];
infoX(X);
sol := [seq(seq(cat(cat('h', i), j) = B[i][j], j = 1 .. 3), i = 1 .. N), c = (B[1]) . (B[2])];
filename := "K15b.csv";
unassign('c');
B := ImportMatrix(filename, source = csv);
N := RowDimension(B) - 1;
B := [seq(Vector[column](B[j, 1 .. -1]), j = 1 .. N + 1)];
B := [seq(Normalize(b, Euclidean), b in B)];
sol := [seq(seq(cat(cat('h', i), j) = B[i][j], j = 1 .. 3), i = 1 .. N), c = (B[1]) . (B[2])];
(B[1]) . (B[2]);
RC := GramSchmidt([B[1], B[2], B[3]], normalized = true);
R := Transpose(Matrix([RC[3], RC[2], RC[1]]));
if Determinant(R) < 0 then
    R := Transpose(Matrix([-RC[3], RC[2], RC[1]]));
end if;
B := map2(Multiply, R, B);
V := coords(B, T, 0.2);
sol := [c = DotProduct(B[1], B[2]), seq(seq(h(i, j) = B[i][j], j = 1 .. 3), i = 1 .. N)];
K := B2K(B, T, signed = true);
print("curvature", evalf(K, 4));
feasible(B);
if 0 < (B[1]) . (B[-1]) then
    oriented := true;
else
    oriented := false;
end if;
oriented;
visTet(V);
H := setupK([0, 0, 0]);
Fix := {op(eval(Fix, s = slide)), s = slide};
cn_fixed := clean_eqs(eval(cn, Fix));
Hf := eval(H, Fix);
Wr := writhe(XC);
eval(c, sol);
printf("total K/Pi %g, total |K| %g, Wr %g, Tt %g\n", add(k, k = K)/3.141592, add(abs(k), k = K), Wr, Wr + Tw(B));
# Experimental
# Curvature evolution
KK := [seq(cat(k, i), i = 1 .. N)];
ode := seq(diff(KK[i](t), t) = -sin(signed_modN(K2omega(KK), i)(t)) - sin(signed_modN(K2omega(KK), i - 1)(t)), i = 1 .. N);
ics := seq(KK[i](0) = K[i], i = 1 .. N);
dsn := dsolve([ode, ics], numeric);
display([seq(listplot(eval([seq(KK[i](t), i = 1 .. N)], dsn(s/10))), s = 1 .. 100)], style = line, thickness = 3, insequence = true);
ode2 := seq(diff(KK[i](t), t) = -tan(signed_modN(KK, i + 1)(t)/2) + tan(signed_modN(KK, i - 1)(t)/2), i = 1 .. N);
dsn2 := dsolve([ode2, ics], numeric);
display([seq(listplot(eval([seq(KK[i](t), i = 1 .. N)], dsn2(s/10))), s = 1 .. 100)], style = line, thickness = 3, insequence = true);
feasible(K2B(eval([seq(KK[i](t), i = 1 .. N)], dsn2(1)), c)[1]);
seq((-tan(signed_modN(K, i + 1)/2) + tan(signed_modN(K, i - 1)/2))/(-sin(signed_modN(K2omega(K), i)) - sin(signed_modN(K2omega(K), i - 1))), i = 1 .. N);
V := K2theta(K);
VV := [seq(cat(v, i), i = 1 .. N)];
ode := seq(diff(signed_modN(VV, i + 1)(t), t) - diff(signed_modN(VV, i)(t), t) = sin((signed_modN(VV, i + 1)(t) + signed_modN(VV, i)(t))/2), i = 1 .. N);
ics := seq(VV[i](0) = V[i], i = 1 .. N);
dsn := dsolve([ode, ics], numeric);
display([seq(listplot(eval([seq(VV[i](t), i = 1 .. N)], dsn(s/10))), s = 1 .. 100)], style = line, thickness = 3, insequence = true);
c := (B[1]) . (B[2]);
Bk, Tk := K2B(theta2K(eval([seq(VV[i](t), i = 1 .. N)], dsn(1))), c);
feasible(Bk);
# Twin / Subdivision / Developable
slide := 0;
cosine := 0;
developable := false;
N := 16;
midaxislen := 1;
oriented := true;
H := setupK(Vector([0, 0, 0]));
width := 0.15;
Fix := {op(eval(Fix, s = slide)), s = slide};
Fix := {op(Fix), h(N/2 + 1, 1) = 0, h(N/2 + 1, 2) = 0, h(N/2 + 1, 3) = -1};
cn_fixed := {op(eval(cn, Fix))};
M := B2T(H);
ExtProd_mid := add(slide*H[i] + M[i], i = 1 .. N/2);
cn := {};
for f in [op(InProd), ExtProd[1], ExtProd[2], ExtProd[3], ExtProd_mid[1], ExtProd_mid[2], ExtProd_mid[3]] do
    if not (f = 0) then cn := {op(cn), f = 0}; end if;
end do;
Hf := eval(H, Fix);
cn_fixed := eval(cn, Fix);
a, sol := find(cn_fixed,-c):print(a);
B := eval(Hf, sol): T := B2T(B): X := B2X(B,T): V := coords(B, T, width): 
visTet(V);
visBand(V);
B := mirror(B);
T := B2T(B);
X := B2X(B, T);
V := coords(B, T, width);
visTet(V);
visBand(V);
Bdiv := subdiv(B, 2);
V := coords(Bdiv, B2T(Bdiv), 0.5);
feasible(Bdiv);
visBand(V);
visBand(coordsTrefoil(Bdiv, B2T(Bdiv), 0.7));

# Developable
# Developable
slide:=1:cosine:=0.2:developable:= true; # for developable, set slide>0
N:=9: midaxislen:=1; oriented := false; H:=setupK(Vector([0,0,0])):width:=0.2:
Fix:={op(eval(Fix,c=cosine)),c=cosine}:
Fix:={op(eval(Fix,s=slide)),s=slide}:
cn_fixed := {op(eval(cn,Fix))}:
Hf:=eval(H,Fix):HA:=add(H[i],i=1..N): 
a, sol := find(cn_fixed,-c):
B := eval(Hf, sol): T := B2T(B): V:=coords(B,T,width):
# display developable band
visBand(V);  

# degenerate example
slide:=0:oriented:=true:width:=0.2:developable:= false:
for i from 1 to N do: 
  P[i]:=Vector([0,0,1]); M[i]:=Vector(evalf([cos((2 *i*Pi)/(N)),sin((2*i*Pi)/(N)),0])):
end do:
V:=coords(P,M,width):
visBand(V);
# Variable segment length
NULL;
H := [seq(Vector([h(i, 1), h(i, 2), h(i, 3)]), i = 1 .. N + 2)];
Nrm := [seq(((H[i]) . (H[i])) - 1, i = 1 .. N)];
ExtProd := add(CrossProduct(H[i], H[i + 1])/abs(sqrt((1 - ((H[i]) . (H[i + 1]))^2)*(1 - ((H[i + 1]) . (H[i + 2]))^2))), i = 1 .. N);
Close := [h(N + 1, 1) + h11, h(N + 1, 2) + h12, h(N + 1, 3) + h13, h(N + 2, 1) + h21, h(N + 2, 2) + h22, h(N + 2, 3) + h23];
C := [op(Nrm), op(Close), ExtProd[1], ExtProd[2], ExtProd[3]];
cn := {};
for f in C do
    if not (f = 0) then cn := {op(cn), f = 0}; end if;
end do;
Fix := fix_global(variableTorsion);
target := mul((H[i]) . (H[i + 1]), i = 1 .. N);
a, sol := op(Minimize(target, cn, h22 = -0.01 .. 1.01, iterationlimit = 10^10, optimalitytolerance = 10^(-35)));
B := eval(Hf, sol);
seq((B[i]) . (B[i + 1]), i = 1 .. N);
# RAG (too slow)
with(RAGMaple);
HasRealSolutions(cn);
lcn := {seq(lhs(z), z in cn)};
#PHCPack
read `run_phc_maple7`:
s := run_phc("/usr/local/bin/phc",[op(lcn)]);
with(RootFinding):
Homotopy(lcn);
Isolate(lcn,[op(indets(lcn))]);
with(Groebner):
IsZeroDimensional(lcn);
gb:=Solve(lcn);
# RegularChains (too slow)
with(RegularChains);
with(ChainTools);
with(MatrixTools);
with(ConstructibleSetTools);
with(ParametricSystemTools);
with(SemiAlgebraicSetTools);
with(FastArithmeticTools);
with(AlgebraicGeometryTools);
R := PolynomialRing([seq(h(i,1),i=3..N),seq(h(i,2),i=2..N),seq(h(i,3),i=2..N-1)]);
Display(R);
lcn := {seq(lhs(z), z in cn)};
tr := RealTriangularize(lcn, R);
Info(tr, R);
cs := Projection(tr, 2, R);
Info(cs, R);
Display(tr, R);
