import argparse
from patterns_model import entrenar_modelo, test_model, GridDraw
from flowers_model import entrenar_modelo_flores, test_flower_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e","--entrenar",action="store_true", help="entrenar modelo y guardar los pesos y bias")
    parser.add_argument("-t","--test",action="store_true", help="probar el modelo")
    parser.add_argument("-d","--dibujar",action="store_true", help="dibujar")
    parser.add_argument("-ef","--entrenar_flores",action="store_true", help="entrenar modelo con flores")
    parser.add_argument("-tf","--test_flores",action="store_true", help="testear modelo con flores")

    args = parser.parse_args()

    if args.entrenar:
        print("==== Entrenando modelo ====")
        return entrenar_modelo()
    elif args.test:
        print("==== Probando modelo ====")
        return test_model()
    elif args.dibujar:
        grid = GridDraw()
        return grid.run()
    elif args.entrenar_flores:
        entrenar_modelo_flores()
    elif args.test_flores:
        test_flower_model()
        return





    pass

if __name__ == '__main__':
    main()