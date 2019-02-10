#ifndef CONTAINER_HEADER
#define CONTAINER_HEADER

#include <vector>

class Container {
    private:
        unsigned int _dimensions;
        double _side_length;
        std::vector<double> sides;

    public:
        Container( unsigned int dimensions, double side_length )
            : _dimensions(dimensions),
              sides(dimensions, side_length),
              _side_length(side_length){
              }

        unsigned int dimensions() { return _dimensions; }
        double side_length() { return _side_length; }
        friend std::ostream& operator<<(std::ostream& s, const Container& box);

};


std::ostream& operator<<(std::ostream& stream, 
                         const Container& box) {
    stream << "{";
    stream << "\"sides\": [";
    std::copy(std::begin(box.sides), std::end(box.sides)-1, 
              std::ostream_iterator<double>(stream, ", "));

    stream << box.sides.back() 
           << "]";

    stream << "}";
    return stream;
}


#endif