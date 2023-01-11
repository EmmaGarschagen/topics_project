#ifndef TIME_H
#define TIME_H

// @sect3{Time class}

// A simple class to store time data. Assume a constant time step
// size.
    class Time {
    public:
        Time(const double time_end,
             const double delta_t)
                :
                timestep(0),
                time_current(0.0),
                time_end(time_end),
                delta_t(delta_t) {}

        virtual ~Time() {}

        double current() const {
            return time_current;
        }

        double end() const {
            return time_end;
        }

        double get_delta_t() const {
            return delta_t;
        }

        unsigned int get_timestep() const {
            return timestep;
        }

        void increment() {
            time_current += delta_t;
            ++timestep;
        }

        void reset(){
            time_current = 0.0;
        }

    private:
        unsigned int timestep;
        double time_current;
        const double time_end;
        const double delta_t;
    };

#endif // TIME_H
