$title GEM-mpr
$eolCom //
Sets
   i 'Nodes'                    / 1*16 /
   n 'Matching sequences'       / n1*n12 /
   l 'Task index'               / l0*l5 /
   ;

Alias (i,j,k,o,d);
Alias (m,n);
Alias (l,v,u);

Set links(i,j) "DA and PT links" /
    1.2
    1.3
    2.3
    2.4
    3.4
    4.5
    4.6
    5.6
    5.7
    6.7
    7.8
    7.9
    8.9
    8.10
    9.10
    10.11
    10.12
    11.12
    11.13
    12.13
    13.14
    13.15
    14.15
    14.16
    15.16
/;
links(i,j) $= links(j,i);

Set occu "RD occupancy" /occu0, occu1/;

// Hyper-network construction
Sets hyper_links(i, j, l, v);

loop((i, j, l, v)$(links(i,j) and ord(l)>=2 and ord(l)=ord(v)), hyper_links(i, j, l, v)=yes);
loop((i, j, l, v)$(ord(i)=ord(j) and ord(v)=ord(l)+1 and ord(l)>=2 and ord(l)<card(l)), hyper_links(i, j, l, v)=yes);

// Demands
Set OD(i, j)    "OD pairs" /
    1 .16
    4 .10
    7 .13
/;

Parameter RD_demand(i, j)   "RD Demand vector"  /
   1.16       40000
   / ;
Parameter RP_demand(i, j)   "RP Demand vector"  /
   4.10       20000
   7.13       20000
   / ;
   
// Matching sequence definitions
Parameter s_pickup(n, l, i, j) 'Matching sequence pickup incidence' /
    n1.l1.4.10 1
    n2.l1.7.13 1
    n3.l1.4.10 1
    n3.l2.4.10 1
    n4.l1.4.10 1
    n4.l3.4.10 1
    n5.l1.7.13 1
    n5.l2.7.13 1
    n6.l1.7.13 1
    n6.l3.7.13 1
    n7.l1.7.13 1
    n7.l3.4.10 1
    n8.l1.4.10 1
    n8.l3.7.13 1
    n9.l1.7.13 1
    n9.l2.4.10 1
    n10.l1.7.13 1
    n10.l2.4.10 1
    n11.l1.4.10 1
    n11.l2.7.13 1
    n12.l1.4.10 1
    n12.l2.7.13 1
/;
s_pickup(n, 'l0', '1', '16') = 1 // Default first task RD departing destination

Parameter s_dropoff(n, l, i, j) 'Matching sequence drop-off incidence' /
    n1.l2.4.10 1
    n2.l2.7.13 1
    n3.l3.4.10 1
    n3.l4.4.10 1
    n4.l2.4.10 1
    n4.l4.4.10 1
    n5.l3.7.13 1
    n5.l4.7.13 1
    n6.l2.7.13 1
    n6.l4.7.13 1
    n7.l2.7.13 1
    n7.l4.4.10 1
    n8.l2.4.10 1
    n8.l4.7.13 1
    n9.l3.4.10 1
    n9.l4.7.13 1
    n10.l3.7.13 1
    n10.l4.4.10 1
    n11.l3.4.10 1
    n11.l4.7.13 1
    n12.l3.7.13 1
    n12.l4.4.10 1
/;
s_dropoff(n, 'l5', '1', '16') = 1; // Default last task RD arriving destination

// Stable matching block
Positive Variable
    x_RD_stable(i,n) 'stable matching RD flow from hat(i)-i_RD with matching sequence n'
    x_RD_DA_stable(i,d) 'stable matching RD flow from hat(i)-i_DA with destination d'
    x_RP_stable(i, d, n, l) 'stable matching RP flow from hat(i)-i_RP with matching sequence n as pickup task l'
    x_RP_PT_stable(i,d) 'stable matching RP flow from hat(i)-i_RP with destination d'
    pi_RD_stable(i, d) 'node potential for RD with OD (i, d)'
    pi_RP_stable(i, d) 'node potential for RP with OD (i, d)'
    
Variable // un-singed variable for equality constraints
    phi_stable(n, l) 'multiplier for the stable matching coupling constraint for RP'
    
    
// Route choice block
Positive Variable
    x_DA_route(i, j, d) 'DA flow on link (i, j) with destination d'
    x_RD_route_occu0(i, j, l, v, n) 'RD flow on RD hyper_link (i, j, l, v) with matching sequence n without RP'
    x_RD_route_occu1(i, j, l, v, n) 'RD flow on RD hyper_link (i, j, l, v) with matching sequence n with RP'
    x_RP_route(i, j, l, v, n, d) 'RP flow on RP hyper_link (i, j, l, v) with matching sequence n'
    x_PT_route(i, j, d) 'PT flow on link (i, j) with destination d'
    pi_DA_route(i, d) 'node potential for DA with OD (i, d)'
    pi_RD_route(i, l, n) 'node potential for RD at node i^l with matching sequence n'
    pi_RP_route(i, l, d, n) 'node potential for RP at node i^l with matching sequence n and destination d'
    pi_PT_route(i, d) 'node potential for PT with OD (i, d)'
    
// Multi-passenger ride-sharing block  
Variable // un-singed variable for equality constraints
    lambda_6_route(i, j, l, v, n) 'RD-RP coupling constraint'
    
// Flow aggregation - Technical construction
Variable
    x_flow(i,j)     'aggregate flow variable';
Equation
    xdef(i,j)       'aggregate flow definition';
xdef(i, j)$links(i,j).. x_flow(i, j) =e= sum(d, x_DA_route(i, j, d)) + sum((l, v, n)$(hyper_links(i, j, l, v)), x_RD_route_occu0(i, j, l, v, n) + x_RD_route_occu1(i, j, l, v, n));

// Link travel time - Technical construction
Variable
    t(i,j)          'travel time on link (i, j)';
Equation
    tdef(i,j)       'travel time definition';
tdef(i, j)$links(i,j).. t(i,j) =e= 5*(1 + 0.15*power(x_flow(i, j)/10000,4))

// Class costs
Variable
    c_DA(i, j, d)               'DA link cost'
    c_RD_occu0(i, j, l, v, n)   'RD without-RD link cost'
    c_RD_occu1(i, j, l, v, n)   'RD with-RD link cost'
    c_RP(i, j, l, v, n)         'RP link cost'
    c_PT(i, j, d)               'PT link cost';
Equation
    c_DA_def(i, j, d)               'DA link cost definition'
    c_RD_occu0_def(i, j, l, v, n)   'RD without-RD link cost definition'
    c_RD_occu1_def(i, j, l, v, n)   'RD with-RD link cost definition'
    c_RP_def(i, j, l, v, n)         'RP link cost definition'
    c_PT_def(i, j, d)               'PT link cost definition';
c_DA_def(i, j, d)$links(i,j)..                                      c_DA(i, j, d)                   =e= t(i,j) + 20;
c_RD_occu0_def(i, j, l, v, n)$hyper_links(i, j, l, v)..             c_RD_occu0(i, j, l, v, n)       =e= t(i,j) + 20; 
c_RD_occu1_def(i, j, l, v, n)$hyper_links(i, j, l, v)..             c_RD_occu1(i, j, l, v, n)       =e= t(i,j) + 0.5*20; 
c_RP_def(i, j, l, v, n)$hyper_links(i, j, l, v)..                   c_RP(i, j, l, v, n)             =e= t(i,j) + 0.5*20;
c_PT_def(i, j, d)$links(i,j)..                                      c_PT(i, j, d)                   =e= t(i,j) + 15;


// Multi-passenger ride-sharing MCP functions
// Ridesharing driver-passenger coupling constraints
Equation
    Func_29(i, j, l, v, n)         'RD-RP coupling constraint for RD';
Func_29(i, j, l, v, n)$(hyper_links(i, j, l, v) and (sum((k, d, u)$(ord(u)<=(ord(v)-1)),s_pickup(n, u, k, d)-s_dropoff(n, u, k, d) ) -1)>0)..
                                                                    (sum((k, d, u)$(ord(u)<=(ord(v)-1)),s_pickup(n, u, k, d)-s_dropoff(n, u, k, d) ) -1)
                                                                    *x_RD_route_occu1(i, j, l, v, n)
                                                                    =e=
                                                                    sum(d, x_RP_route(i, j, l, v, n, d));
// Route choice MCP functions
Equation
    Func_33(i, j, d)            'DA route choice'
    Func_34(i, j, l, v, n)      'RD without RP route choice'
    Func_35(i, j, l, v, n)      'RD with RP route choice'
    Func_36(i, j, l, v, n, d)   'RP route choice'
    Func_37(i, j, d)            'PT route choice';
Func_33(i, j, d)$links(i, j)..                                      pi_DA_route(j, d) + c_DA(i, j, d) =g= pi_DA_route(i, d);
Func_34(i, j, l, v, n)$hyper_links(i, j, l, v)..                    pi_RD_route(j, v, n) + c_RD_occu0(i, j, l, v, n)
                                                                    =g=
                                                                    pi_RD_route(i, l, n);
Func_35(i, j, l, v, n)$hyper_links(i, j, l, v)..                    pi_RD_route(j, v, n) + c_RD_occu1(i, j, l, v, n)
                                                                    -(sum((k, d, u)$(ord(u)<=(ord(v)-1)),s_pickup(n, u, k, d)-s_dropoff(n, u, k, d) ) -1)*lambda_6_route(i, j, l, v, n)
                                                                    =g=
                                                                    pi_RD_route(i, l, n);
Func_36(i, j, l, v, n, d)$hyper_links(i, j, l, v)..                 pi_RP_route(j, v, d, n) + c_RP(i, j, l, v, n)
                                                                    +lambda_6_route(i, j, l, v, n)
                                                                    =g=
                                                                    pi_RP_route(i, l, d, n);
Func_37(i, j, d)$links(i, j)..                                      pi_PT_route(j, d) + c_PT(i, j, d) =g= pi_PT_route(i, d);


// RD stable matching MCP functions
Equation
    Func_38a(i, n)               'RD stable matching sequence choice'
    Func_38b(i, d)               'RD stable matching quiting choice'
    Func_38c(i, d)               'RD stable matching conservation'
    Func_38d(n, l)               'RD stable matching RD-RP coupling';
Func_38a(i, n)$(sum(j , s_pickup(n, 'l0', i, j)) = 1)..             pi_RD_route(i, 'l1', n) +
                                                                    sum(l$(ord(l)>=2 and (ord(l) < card(l))  and sum((j, d), s_pickup(n, l, j, d)) = 1) , phi_stable(n, l))
                                                                    =g=
                                                                    sum(d$(s_pickup(n, 'l0', i, d) = 1), pi_RD_stable(i, d)); //Corresponding pi_i^d for matching sequence n
Func_38b(i, d)..                                                    pi_DA_route(i, d)
                                                                    =g=
                                                                    pi_RD_stable(i, d);
Func_38c(i, d)..                                                    sum(n$(s_pickup(n, 'l0', i, d) = 1), x_RD_stable(i, n))
                                                                    + x_RD_DA_stable(i,d)
                                                                    =g=
                                                                    RD_demand(i, d);
Func_38d(n, l)$(ord(l)>=2 and (ord(l) < card(l)) and sum((j, d), s_pickup(n, l, j, d)) = 1)..
                                                                    sum(i$(sum(d, s_pickup(n, 'l0', i, d)) = 1) , x_RD_stable(i,n))
                                                                    =e=
                                                                    sum((j, d)$(s_pickup(n, l, j, d)=1), x_RP_stable(j, d, n, l));

// RP stable matching MCP functions
Equation
    Func_39a(i, d, n, l)         'RP stable matching sequence choice'
    Func_39b(i, d)               'RP stable matching quiting choice'
    Func_39c(i, d)               'RP stable matching conservation'
    Func_39d(n, l)               'RP stable matching RD-RP coupling';
Func_39a(i, d, n, l)$(ord(l)>=2 and (ord(l) < card(l)) and s_pickup(n, l, i, d) = 1)..
                                                                    pi_RP_route(i, l, d, n) -
                                                                    phi_stable(n, l)
                                                                    =g=
                                                                    pi_RP_stable(i, d);
Func_39b(i, d)..                                                    pi_PT_route(i, d)
                                                                    =g=
                                                                    pi_RP_stable(i, d);
Func_39c(i, d)..                                                    sum((n, l)$(ord(l)>=2 and (ord(l) < card(l)) and s_pickup(n, l, i, d) = 1), x_RP_stable(i, d, n, l))
                                                                    + x_RP_PT_stable(i,d)
                                                                    =g=
                                                                    RP_demand(i, d);
Func_39d(n, l)$(ord(l)>=2 and (ord(l) < card(l)) and sum((j, d), s_pickup(n, l, j, d)) = 1)..
                                                                    sum(i$(sum(d, s_pickup(n, 'l0', i, d)) = 1) , x_RD_stable(i,n))
                                                                    =e=
                                                                    sum((j, d)$(s_pickup(n, l, j, d)=1), x_RP_stable(j, d, n, l));

// Route choice flow conservation MCP functions
Equation
    Func_40a(i, d)              'DA flow conservation'
    Func_40b(i, l, n)           'RD flow conservation'
    Func_40c(i, l, d, n)        'RP flow conservation'
    Func_40d(i, d)              'PT flow conservation';
Func_40a(i, d)..                                                    sum(j$links(i,j), x_DA_route(i, j, d))
                                                                    =g=
                                                                    x_RD_DA_stable(i, d)
                                                                    +sum(k$links(k,i), x_DA_route(k, i, d));
Func_40b(i, l, n)$(ord(l)>=2)..                                     sum((j, v)$hyper_links(i, j, l, v), x_RD_route_occu0(i, j, l, v, n))
                                                                    +sum((j, v)$hyper_links(i, j, l, v), x_RD_route_occu1(i, j, l, v, n))
                                                                    =g=
                                                                    x_RD_stable(i,n)$(ord(l)=2)
                                                                    +sum((k, u)$hyper_links(k, i, u, l), x_RD_route_occu0(k, i, u, l, n))
                                                                    +sum((k, u)$hyper_links(k, i, u, l), x_RD_route_occu1(k, i, u, l, n));
Func_40c(i, l, d, n)$(ord(l)>=2)..                                  sum((j, v)$hyper_links(i, j, l, v), x_RP_route(i, j, l, v, n, d))
                                                                    =g=
                                                                    x_RP_stable(i, d, n, l)
                                                                    +sum((k, u)$hyper_links(k, i, u, l), x_RP_route(k, i, u, l, n, d));
Func_40d(i, d)..                                                    sum(j$links(i,j), x_PT_route(i, j, d))
                                                                    =g=
                                                                    x_RP_PT_stable(i, d)
                                                                    +sum(k$links(k,i), x_PT_route(k, i, d));                                                                    
// Assign zero-cost to virtual links                                                                   
t.fx(i,i) = 0;

// RD-RP coupling constraint - without RP
x_RP_route.fx(i, j, l, v, n, d)$(hyper_links(i, j, l, v) and sum((k, o, u)$(ord(u)<=(ord(v)-1)),s_pickup(n, u, k, o)-s_dropoff(n, u, k, o) ) -1 = 0) = 0;

// RD with and without RP conservation at intermediate nodes
x_RD_route_occu1.fx(i, j, l, v, n)$(hyper_links(i, j, l, v) and sum((k, o, u)$(ord(u)<=(ord(v)-1)),s_pickup(n, u, k, o)-s_dropoff(n, u, k, o) ) -1 = 0)     = 0;
x_RD_route_occu0.fx(i, j, l, v, n)$(hyper_links(i, j, l, v) and sum((k, o, u)$(ord(u)<=(ord(v)-1)),s_pickup(n, u, k, o)-s_dropoff(n, u, k, o) ) -1 > 0)     = 0;
lambda_6_route.fx(i, j, l, v, n)$(hyper_links(i, j, l, v) and (sum((k, d, u)$(ord(u)<=(ord(v)-1)),s_pickup(n, u, k, d)-s_dropoff(n, u, k, d) ) -1)=0)    = 0;

// Matching sequence intra-task constraint
x_RD_route_occu0.fx(i, i, l, l+1, n)$(sum(d, s_pickup(n, l, i, d)) + sum(k, s_dropoff(n, l, k, i)) = 0) = 0;
x_RD_route_occu1.fx(i, i, l, l+1, n)$(sum(d, s_pickup(n, l, i, d)) + sum(k, s_dropoff(n, l, k, i)) = 0) = 0;
                                                                    
// Matching sequence inter-task constraint
x_RD_route_occu0.fx(i, j, l, l, n)$(hyper_links(i, j, l, l) and sum(d, s_pickup(n, l, i, d)) + sum(k, s_dropoff(n, l, k, i)) > 0) = 0;
x_RD_route_occu1.fx(i, j, l, l, n)$(hyper_links(i, j, l, l) and sum(d, s_pickup(n, l, i, d)) + sum(k, s_dropoff(n, l, k, i)) > 0) = 0;

// Node potential at destination set as 0
pi_DA_route.fx(d, d)                                                            = 0;
pi_RD_route.fx(d, l, n)$(ord(l)=card(l) and sum(k, s_dropoff(n, l, k, d))=1)    = 0;
pi_RP_route.fx(d, l, d, n)                                                      = 0;
pi_PT_route.fx(d, d)                                                            = 0;

// Flow termination at destination
x_DA_route.fx(d, j, d)                                                                  = 0;
x_RP_route.fx(d, j, l, v, n, d)                                                         = 0;
x_RD_route_occu0.fx(d, j, l, v, n)$(ord(l)=card(l) and sum(k, s_dropoff(n, l, k, d))=1) = 0;
x_RD_route_occu1.fx(d, j, l, v, n)$(ord(l)=card(l) and sum(k, s_dropoff(n, l, k, d))=1) = 0;
x_PT_route.fx(d, j, d)                                                                  = 0;

// Fix unmatched/free variables
lambda_6_route.fx(i, j, l, v, n)$(not hyper_links(i, j, l, v))      = 0;
x_RD_stable.fx(i,n)$(sum(d, s_pickup(n, 'l0', i, d))=0)             = 0;
x_RP_stable.fx(i, d, n, l)$(s_pickup(n, l, i, d)=0)                 = 0;

Model
   GEM  'mcp formulation' /
   xdef.x_flow, tdef.t, c_DA_def.c_DA, c_RD_occu0_def.c_RD_occu0, c_RD_occu1_def.c_RD_occu1, c_RP_def.c_RP, c_PT_def.c_PT,
   Func_29.lambda_6_route
   Func_33.x_DA_route, Func_34.x_RD_route_occu0, Func_35.x_RD_route_occu1, Func_36.x_RP_route, Func_37.x_PT_route,
   Func_38a.x_RD_stable, Func_38b.x_RD_DA_stable, Func_38c.pi_RD_stable, Func_38d.phi_stable,
   Func_39a.x_RP_stable, Func_39b.x_RP_PT_stable, Func_39c.pi_RP_stable,
   Func_40a.pi_DA_route, Func_40b.pi_RD_route, Func_40c.pi_RP_route, Func_40d.pi_PT_route
 /;

option mcp = path;
solve GEM using mcp; 