<script setup lang="ts">

import {onBeforeMount, onMounted, ref} from "vue";
import juegoProlog from './juego.pl?raw'

const prolog = window.pl.create()


interface Jugada {
    emoji: string;
    nombre: string;
    color: string;
    key: string
}


const DELAY = 150
const NUMERO_MAXIMO_TIROS = 15

const jugadas_validas: readonly Jugada[] = Object.freeze([
    {
        nombre: 'Piedra',
        emoji: '✊',
        color: 'bg-gray-200 hover:border-gray-500',
        key: 'a'
    },
    {
        nombre: 'Papel',
        emoji: '📄',
        color: 'bg-orange-100 hover:border-orange-500',
        key: 's',
    },
    {
        nombre: 'Tijeras',
        emoji: '✂️',
        color: 'bg-red-100 hover:border-red-500',
        key: 'd'
    },
    {
        nombre: 'Lagarto',
        emoji: '🦎',
        color: 'bg-emerald-100 hover:border-emerald-500',
        key: 'f'
    },
    {
        nombre: 'Spock',
        emoji: '🖖',
        color: 'bg-blue-100 hover:border-blue-500',
        key: 'g'
    },
])

const esta_jugando = ref<boolean>(false)

const jugadaActual = ref<Jugada>()
const jugadaRival = ref<Jugada>()

const respuesta = ref<string>()

const determinarJugada = () => {
    if (!jugadaActual.value || !jugadaRival.value)
        return
    const jugadaUno = jugadaActual.value.nombre.toLowerCase()
    const jugadaDos = jugadaRival.value.nombre.toLowerCase()
    prolog.query(`resultado(${jugadaUno},${jugadaDos},R).`, {
        success(){
            prolog.answer({
                success({links}: any){
                    const {R} = links
                    const res = R.id
                    respuesta.value = res
                }
            })
        },
        error(){
            console.error('error parsing prolog')
        }
    })
}


const tirarJugadaRival = (numero_tiro: number = 0) => {
    const index = Math.floor(Math.random() * jugadas_validas.length)
    jugadaRival.value = jugadas_validas[index]

    if (numero_tiro > NUMERO_MAXIMO_TIROS){
        determinarJugada()
        esta_jugando.value = false
        return; // detenemos el caso
    }
    setTimeout(() => {
        tirarJugadaRival(numero_tiro + 1)
    },DELAY)
}

const elegirJugada = (jugada: Jugada) => {
    if (esta_jugando.value)
        return // no lanzar ningun evento mientras se juega
    jugadaActual.value = jugada
    esta_jugando.value = true
    tirarJugadaRival()
}

onBeforeMount(() => {
    prolog.consult(juegoProlog,{
        error(){
            console.warn('error al cargar el programa prolog')
        },
        success(){
            console.info('prolog loaded successfully')
        }
    })

})

onMounted(() => {
    for(const jugada of jugadas_validas){
        const event = (e: KeyboardEvent) => {
            if (e.key === jugada.key)
                elegirJugada(jugada)
        }
        document.addEventListener('keydown',event)
    }

})




</script>

<template>
  <div class="bg-slate-100 h-screen">
      <div class="container mx-auto h-screen flex flex-col">
          <div>
              <h1 class="text-4xl text-center py-10 font-semibold italic">Piedra, Papel, Tijera, Lagarto, Spoke</h1>
          </div>
          <div class="flex-grow flex flex-col justify-center">
              <div>
                  <div v-if="jugadaActual && jugadaRival" class= "grid grid-cols-3">
                      <div class="text-center">
                          <span class="block mb-4 text-5xl">{{ jugadaActual.emoji }}</span>
                          <h2 class="text-2xl">{{ jugadaActual.nombre }}</h2>
                      </div>
                      <span class="text-4xl italic bold text-center">VS</span>
                      <div class="text-center">
                          <span class="block mb-4 text-5xl">{{ jugadaRival.emoji }}</span>
                          <h2 class="text-2xl">{{ jugadaRival.nombre }}</h2>
                      </div>
                  </div>
                  <div>
                      <h1 class="text-center mt-4 text-4xl">{{ respuesta }}</h1>
                  </div>
              </div>
          </div>
          <div class="grid flex-shrink-0 mb-[5rem] grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              <div :class="jugada.color"
                   @click.prevent="elegirJugada(jugada)"
                   class="p-5 text-lg rounded shadow border-b-5 cursor-pointer border-gray-200 transition-all"
                   v-for="jugada in jugadas_validas">
                  {{ jugada.emoji }}
                  {{ jugada.nombre }}
                  <span class="px-1 rounded border border-gray-300 text-stone-500">
                      {{ jugada.key }}
                  </span>
              </div>
          </div>
      </div>

  </div>
</template>

<style scoped>

</style>
