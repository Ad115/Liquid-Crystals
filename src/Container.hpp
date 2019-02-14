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

        unsigned int dimensions() { return _side_lengths.size(); }
        double side_length() { return _side_lengths[0]; }
        const std::vector<double>& side_lengths() const { return _side_lengths; }
        friend std::ostream& operator<<(std::ostream&, const Container&);

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