static void spinsleep() {
  volatile double x = 2;
  for(int i = 0; i < 1<<27; i++) x = x * 2 + 3;
}

static void caller() {
  spinsleep();
}

int main() {
  spinsleep(); caller();
}
