class NexusUniversalLanguage:
    """
    Intérprete de un lenguaje espacial declarativo y reflexivo.
    Permite crear comandos nuevos en tiempo de ejecución (metaprogramación)
    y controlar tanto AR como VR y pines de hardware electrónico.
    """
    def __init__(self):
        self.world_objects = {}
        self.learned_macros = {}
        self.reality_mode = "MIXED_50_50"

        # Diccionario base de instrucciones universales
        self.instruction_table = {
            "MODO_REALIDAD": self._set_reality,
            "CREAR_HOLOGRAMA": self._spawn_entity,
            "VINCULAR_MANO": self._bind_hand_vector,
            "APRENDER_COMANDO": self._define_self_modifying_macro,
            "HARDWARE_PIN": self._write_electronic_device,
        }

    def _set_reality(self, args):
        self.reality_mode = args[0]
        print(f"[NEXUS-LANG] Transición óptica activada -> Modo: {self.reality_mode}")

    def _spawn_entity(self, args):
        name, obj_type, x, y, z = args[0], args[1], float(args[2]), float(args[3]), float(args[4])
        self.world_objects[name] = {"type": obj_type, "pos": [x, y, z], "bound_to_hand": False}
        print(f"[NEXUS-LANG] Objeto '{name}' ({obj_type}) materializado en coordenadas ({x}, {y}, {z})")

    def _bind_hand_vector(self, args):
        obj_name, finger_joint = args[0], args[1]
        if obj_name in self.world_objects:
            self.world_objects[obj_name]["bound_to_hand"] = finger_joint
            print(f"[NEXUS-LANG] Objeto '{obj_name}' anclado al vector de la mano: {finger_joint}")

    def _write_electronic_device(self, args):
        pin, state = int(args[0]), args[1]
        print(f"[HARDWARE-OS] Señal eléctrica enviada al Pin Físico #{pin} -> Estado: {state}")

    def _define_self_modifying_macro(self, args):
        """Permite que el lenguaje cree nuevas palabras clave por sí mismo."""
        macro_name = args[0]
        raw_subcommands = " ".join(args[1:]).split(";")
        self.learned_macros[macro_name] = [cmd.strip() for cmd in raw_subcommands if cmd.strip()]
        print(f"[METAPROGRAMACIÓN] El lenguaje ha aprendido una nueva palabra clave: '{macro_name}'")

    def execute_script(self, script_text: str):
        for line in script_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            cmd, args = tokens[0].upper(), tokens[1:]

            if cmd in self.instruction_table:
                self.instruction_table[cmd](args)
            elif cmd in self.learned_macros:
                for sub_line in self.learned_macros[cmd]:
                    self.execute_script(sub_line)
            else:
                print(f"[ERROR SINTÁCTICO] Comando desconocido: {cmd}")

# Ejemplo de un programa escrito en tu nuevo lenguaje universal:
if __name__ == "__main__":
    sample_code = """
    # 1. Configuramos fusión entre Realidad Aumentada y Virtual
    MODO_REALIDAD HIBRIDO_CUANTICO_AR_VR

    # 2. Creamos un objeto 3D flotando frente al usuario
    CREAR_HOLOGRAMA NucleoEnergia ESFERA_PLASMA 0.0 1.5 -0.5

    # 3. Lo conectamos al vector de la punta del dedo índice derecho
    VINCULAR_MANO NucleoEnergia DEDO_INDICE_PUNTA

    # 4. El lenguaje se modifica a sí mismo creando un comando nuevo
    APRENDER_COMANDO ACTIVAR_PORTAL MODO_REALIDAD VR_TOTAL ; HARDWARE_PIN 13 ENCENDIDO

    # 5. Ejecutamos el comando que el propio código acaba de inventar
    ACTIVAR_PORTAL
    """
    interpreter = NexusUniversalLanguage()
    interpreter.execute_script(sample_code)
