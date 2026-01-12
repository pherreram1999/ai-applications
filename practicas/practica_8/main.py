#!/usr/bin/python3

import argparse
from patterns_model import entrenar_modelo, test_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e","--entrenar",action="store_true", help="entrenar modelo y guardar los pesos y bias")
    parser.add_argument("-t","--test",action="store_true", help="probar el modelo")

    args = parser.parse_args()

    if args.entrenar:
        print("==== Entrenando modelo ====")
        return entrenar_modelo()
    elif args.test:
        print("==== Probando modelo ====")
        return test_model()

if __name__ == '__main__':
    main()
