import numpy as np
import matplotlib.pyplot as plt
import random
import math
import matplotlib.lines as lines

w = 0.5  # Intertia weight to prevent velocities to come too large (between 0.4 and 0.9)
c1 = 2  # C1 gives importance of PERSONNAL best value, scaling coefficient on the social component
c2 = 2  # C2 gives importance of SWARM (SOCIAL) best value, scaling coefficient on the cognitive component
iteration = 5
nb_particles = 3

x_final = random.random() * 10
y_final = random.random() * 10
x = float
y = float

velocity = []
new_velocity = []
x_coordinates = []
y_coordinates = []
x_coordinates_new = []
y_coordinates_new = []

particle = []


def particles(x, y):
    dist = []
    index = 0
    max_range = 10
    for i in range(nb_particles):
        x_coordinates.append(random.random() * 10)  #On set up la coordonnee x
        y_coordinates.append(random.random() * 10)  #On set up la coordonnee y
        velocity.append(0.4 * random.random() * 10) #On set up la velocite
        dist.append(calculateDistance(x_coordinates[i], y_coordinates[i], x_final, y_final))

        plt.plot(x_coordinates, y_coordinates, 'ro') #On affiche

        for j in range(len(dist)):
            if dist[j] < max_range:
                max_range = dist[j]
                index = j

        closet_x = x_coordinates[index]
        closet_y = y_coordinates[index]

        plt.plot(closet_x, closet_y, 'mo')
    return


def move(x, y):
    for j in range(10):
        for i in range(nb_particles):
            new_velocity[i] = (w * velocity[i]) + (c1 * random.random())
            a = x_coordinates[i] + new_velocity[i] #La prochaine position est egale a l'ancienne plus la velocite de la particle
            b = y_coordinates[i] + new_velocity[i] * random.random() #La prochaine position est egale a l'ancienne plus la velocite de la particle
            plt.plot(a, b, 'yo')
            plt.plot([x_coordinates[i],a], [y_coordinates[i],b ], 'k-')
            x_coordinates[i] = a
            y_coordinates[i] = b
    return

def calculateDistance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

particles(x,y)
move(x,y)


# randomPoints(x, y)
plt.plot(x_final, y_final, 'bo')
plt.show()
