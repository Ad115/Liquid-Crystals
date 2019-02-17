#ifndef CONTAINER_HEADER
#define CONTAINER_HEADER

#include <vector>


class Container {
    private:
        std::vector<double> _side_lengths;

    public:
        Container( unsigned int dimensions, double side_length )
            : _side_lengths(dimensions, side_length)
              {}

        unsigned int dimensions() const { return _side_lengths.size(); }
        double side_length() const { return _side_lengths[0]; }
        const std::vector<double>& side_lengths() const { return _side_lengths; }
        friend std::ostream& operator<<(std::ostream&, const Container&);

        Vector apply_boundary_conditions(const Vector& position) const;
        Vector minimum_image(const Vector& p1, const Vector& p2) const;

};


class PeriodicBoundaryBox : public Container {
    public: 

    PeriodicBoundaryBox( unsigned int dimensions, double side_length )
            : Container(dimensions, side_length)
              {}

    Vector apply_boundary_conditions(const Vector& position) const {
        Vector new_pos(position);
        for (int i=0; i<new_pos.dimensions();i++){
            new_pos[i] -= (new_pos[i] > side_length()) * side_length();
            new_pos[i] += (new_pos[i] < 0) * side_length();
        }
        return new_pos;
    }

    Vector minimum_image(const Vector& r1, const Vector& r2) const { /*
        * Get the distance to the minimum image.
        */
        double half_length = side_length()/2;
        Vector dr = r1 - r2;

        for(int D=0; D<dimensions(); D++) {

            if (dr[D] <= -half_length) {
                dr[D] += side_length();

            } else if (dr[D] > +half_length) {
                dr[D] -= side_length();
            }
        }
        return dr;
    }
};

std::ostream& operator<<(std::ostream& stream, 
                         const Container& box) {
    stream << "{";
    stream << "\"side_lengths\": [";

    auto sides = box.side_lengths();
    std::copy(std::begin(sides), std::end(sides)-1, 
              std::ostream_iterator<double>(stream, ", "));

    stream << sides.back() 
           << "]";

    stream << "}";
    return stream;
}


#endif
