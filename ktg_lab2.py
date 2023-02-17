import math

h = 1.6E-3
er = 4.4
tand = 0.0018
t = 35E-6
fr = 2.45E+9
c = 3E+8

w = c/(2*fr)*math.sqrt(2/(er+1))

er_eff = (er+1)/2 + (er-1)/2*1/math.sqrt(1+12*h/w)

delta_L = 0.412*h*((er_eff+0.3)*(w/h+0.264)/((er_eff-0.258)*(w/h+0.8)))

L = w/1.5
L_e = L+2*delta_L
lambda0 = c/fr#/math.sqrt(er))
Rin = 50 #in order for antenna to be matched with TL
k0 = math.sqrt((math.pi/w)**2)
G1 = w/(120*lambda0)*(1-(k0*h)**2/24)
x0 = math.acos(math.sqrt(Rin*2*G1))*L/math.pi
# print(L)
# print(delta_L)
print(f"Length of antenna: {L*1000} mm")
print(f"width of antenna: {w*1000} mm")
print(f"effective permiability: {er_eff}")
print(f"Electrical length adjusted for fringing: {L_e*1000}mm")
print(f"x0 = {x0*1000} mm")
print(f"G1= {G1}")
print(f"deltaL = {delta_L*1000}mm")
