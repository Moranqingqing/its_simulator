import sumolib
import xml.etree.ElementTree as ET

ADDITIONAL_SUFFIX = ".add.xml"

class AdditionalFile:

    def __init__(self, net, name="additional", edgeData_attrib=None, loop_detectors=False,
                 vType_attrib=None, **kwargs):

        """
        :param net: SUMONet instance
        :param name: name of the .add.xml file
        :param edge_data_attrib: dictionary of attributes for the edge_data element
        :param loop_detectors: add loop_detector elements
        :param vType_attrib: vehicle attributes
        :param kwargs: edgeData_attrib include id, file, freq, trackVehicles
        """

        self.net = net
        self.path = self.net.path
        self.additional_file = self.path + name + ADDITIONAL_SUFFIX

        with open(self.additional_file, 'w') as f:
            sumolib.writeXMLHeader(f, "$Id$", "additional")

            if edgeData_attrib is not None:
                edge = ET.Element("edgeData", attrib=edgeData_attrib)
                f.write('    ' + ET.tostring(edge, "unicode") + '\n')
                #output_file = kwargs['edge_output_file']
                #f.write('    <edgeData id="%s" file="%s" freq="%s" trackVehicles="%s" %s/>\n' %
                        #("edge_output", output_file, kwargs["frequency"], kwargs["track"], ""))

            if loop_detectors:
                for edge in self.net.edges:
                    length = edge.getLength()
                    pos = str(round(length / 2, 2))
                    for lane in edge.getLanes():
                        laneID = lane.getID()
                        detectorID = laneID + '_detector'
                        filename = self.path + kwargs["directory"] + detectorID + '_output.xml'

                        if kwargs["detector_type"] == "instant":
                            f.write('    <instantInductionLoop id="%s" lane="%s" pos="%s" file="%s" %s/>\n' %
                                    (detectorID, laneID, pos, filename, ""))

                        elif kwargs["detector_type"] == "e1":
                            f.write('    <inductionLoop id="%s" lane="%s" pos="%s" frequency="%s" file="%s" %s/>\n' %
                                    (detectorID, laneID, pos, kwargs["frequency"], filename, ""))

            if vType_attrib is not None:
                veh = ET.Element("vType", attrib=kwargs["vType_attrib"])
                f.write(ET.tostring(veh, "unicode") + '\n')

            f.write("</additional>")



