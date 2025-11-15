% jugadas validas
jugada_valida(piedra).
jugada_valida(papel).
jugada_valida(tijera).
jugada_valida(lagarto).
jugada_valida(spock).

% reglas del juego

gana(piedra, lagarto).
gana(piedra, tijera).
gana(papel, piedra).
gana(papel, spock).
gana(tijera, papel).
gana(tijera, lagarto).
gana(lagarto, papel).
gana(lagarto, spock).
gana(spock, tijera).
gana(spock, piedra).

% funciones del juego

resultado(X, X, Empate) :- jugada_valida(X).
resultado(X, Y, 'Gana Jugador 1') :- gana(X, Y).
resultado(X, Y, 'Gana Jugador 2') :- gana(Y, X).