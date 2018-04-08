#ÖNEMLİ: kodun çalışması için resminizi 'image.jpg' olarak kodun bulunduğu dosyaya eklemelisiniz
import cv2
#import glob
import numpy as np
#import os
import random
import operator
import copy





def Fitness(image):
    err = np.sum((img.astype("float") - image.astype("float")) ** 2)
    err /= float(img.shape[0] * img.shape[1]*255*3)
    return(100-err)

def GenerateSquer():
    squer = [0,0,0,0,0,0,0]#[x,y,width,height,R,G,B]
    squer[0] = int(random.random()*img.shape[0])
    squer[1] = int(random.random()*img.shape[1])
    squer[2] = int((random.random()*(img.shape[0]-squer[0]))*((100 - Fitness(GI))/100)+1)#we make sure that the squers
    squer[3] = int((random.random()*(img.shape[1]-squer[1]))*((100 - Fitness(GI))/100)+1)#will fit in our image
    #as we get closer to find the image we start to make individuals 
    #with smaller squers wich means makeing smaller changes
    squer[4] = int(random.random()*255)
    squer[5] = int(random.random()*255)
    squer[6] = int(random.random()*255)
    return(squer)

def GenerateIndividual(k):
    individual = []
    for x in range(k):
        individual.append(GenerateSquer())
    return(individual)

def GeneratePopulation(PopulationSize,k):
    population = []
    for x in range(PopulationSize):
        population.append(GenerateIndividual(k))
    return(population)

def GenerateImage(individual):
    GeneratedImage = copy.deepcopy(GI)
    for squer in individual:
        for x in range(squer[0],squer[0]+squer[2]):
            for y in range(squer[1],squer[1]+squer[3]):
                GeneratedImage[x,y,0] = squer[4]
                GeneratedImage[x,y,1] = squer[5]
                GeneratedImage[x,y,2] = squer[6]
    return(GeneratedImage)

def SortBasedOnFitness(population):
    Fitnesses = []
    for index, individual in enumerate(population):
        Fitnesses.append([Fitness(GenerateImage(individual)),index])
    Fitnesses.sort(key = operator.itemgetter(0))
    SortedPopulation = list(population)
    for index, individual in enumerate(population):
        SortedPopulation[index] = population[Fitnesses[index][1]]
    return(SortedPopulation)

def CalculateChances(population):
    summ = 0
    for index, individual in enumerate(population):
        summ += (index + 1)
        #index starts from 0 so we have to add 1 so 
        #the first individual will have a chance
        size = index + 1
        #size = size of the population when the for loop ends
    Chances = [0] * size
    for x in range(size):
        Chances[x] = (x+1) / summ
    return(Chances)

def SelectFromPopulation(population, count):
    Survivors = []
    SortedPopulation = SortBasedOnFitness(population)
    for x in range(count):
        Chances = CalculateChances(SortedPopulation)
        LuckyNumber = random.random()
        #LuckyNumber is a number between 0 and 1 that chooses the survivors 
        Luck = Chances[0]
        i = 0
        while LuckyNumber > Luck:
            i += 1
            Luck += Chances[i]
        Survivors.append(SortedPopulation[i])
        del SortedPopulation[i]
        #deletes the Survived individual so it 
        #won't be selected again in the future
    return(Survivors)

def Breed(IndividualAlpha, IndividualBeta):
    Child = []
    CutSpot = int(random.random()*len(IndividualAlpha))
    for x in range(CutSpot):
        Child.append(IndividualAlpha[x])
    if random.random() < 0.000001 or CutSpot > len(IndividualBeta):
        for x in range(int(random.random()*len(IndividualBeta)),len(IndividualBeta)):
            Child.append(IndividualBeta[x])
        #We create a Child the first part of the Child is from IndiviualAlpha
        #and the second part is from IndividualBeta but we cut our two parrent
        #individuals from different indexes so we can make new individuals with
        #different lengths
    else:
        for x in range(CutSpot, len(IndividualBeta)):
            Child.append(IndividualBeta[x])
    return(Child)

def Mutate(Individual):
    Individual[int(random.random()*len(Individual))] = GenerateSquer()
    return(Individual)

def GenerateNextGeneration(population):
    NextGen = SelectFromPopulation(population, int(len(population)*(0.4)))
    while len(NextGen) < len(population):
        Alpha = int(random.random() * len(NextGen))#First parent of the child
        Beta = int(random.random() * len(NextGen))#Second parent of the child
        while Alpha == Beta:#Both parents can't be the same individual
            Beta = int(random.random() * len(NextGen))
        NextGen.append(Breed(NextGen[Alpha], NextGen[Beta]))
    BreedChance = 0.000001#BreedChance is the probability of the next breeding
    Chance = random.random()
    #Chance will determine rather the next breeding will happen or not
    while Chance < BreedChance:
        Alpha = int(random.random() * len(NextGen))#First parent of the child
        Beta = int(random.random() * len(NextGen))#Second parent of the child
        while Alpha == Beta:#Both parents can't be the same individual
            Beta = int(random.random() * len(NextGen))
        NextGen.append(Breed(NextGen[Alpha], NextGen[Beta]))
        BreedChance -= random.random()
        #we reduce the breed chance each time so we won't have 
        #infinit number of individuals in our next generation
        Chance = random.random()
    MutationChance = 0.0175#Probability of the next mutation
    Chance = random.random()
    #this time Chance will determine rather the next mutation will happen or not
    NextGen = SortBasedOnFitness(NextGen)
    while Chance < MutationChance:
        rand = int(random.random()*(len(NextGen)-1))
        NextGen[rand] = Mutate(NextGen[rand])
        #never mutate the most fitted individual
        MutationChance -= random.random()
        #we reduce the  MutationChance each time 
        #so we won't change our individuals too much
        Chance = random.random()
    return(NextGen)


filename = "image.jpg"
img=cv2.imread(filename)#+input('image name...'))#Our orginal image
img = cv2.resize(img, (int(img.shape[1]*(150/img.shape[0])), 150))
GI = cv2.imread("image.jpgGeneratedImage.jpg")#GI=Generated Image

Population = GeneratePopulation(15,5)
    #FP = First Population, GeneratePopulation(Population size, Shapes in yeach population)
    #cv2.imshow('GI',GI)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('GeneratedImage.jpg',GI)
    #input(Fitness(GI))
GenerationCount = 1
cv2.imwrite(str(filename) + 'Resized.jpg',img)
while Fitness(GenerateImage(SortBasedOnFitness(Population)[len(Population)-1])) < 99: #and GenerationCount < 500000:
    Population = GenerateNextGeneration(Population)
    #    input(Fitness(GenerateImage(Population[len(Population)-1])))
    #    input(Fitness(GI))
    if Fitness(GenerateImage(Population[len(Population)-1])) > Fitness(GI):
        GI = GenerateImage(SortBasedOnFitness(Population)[len(Population)-1])
    cv2.imwrite(str(filename)+'GeneratedImage.jpg',GI)
    print('length', len(Population))
    GenerationCount += 1
    print('------------------------------------------------------------------')
    print('GENERATION ',GenerationCount)
    print('FITNESS: ', Fitness(GI))#Fitness(GenerateImage(SortBasedOnFitness(Population)[len(SortBasedOnFitness(Population))-1])))




#LeastFitted = GenerateImage(SortedPopulation[0])
#LeastFitted = cv2.resize(LeastFitted, (int(img.shape[1]*(400/img.shape[0])), 400))
#MostFitted = GenerateImage(SortedPopulation[len(SortedPopulation)-1])
#MostFitted = cv2.resize(MostFitted, (int(img.shape[1]*(400/img.shape[0])), 400))
#cv2.imshow('leastFitted',GenerateImage(SortedPopulation[0]))
#cv2.imshow('mostFitted',GenerateImage(SortedPopulation[len(SortedPopulation)-1]))
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
